"""Build pinned DUCC FFT branches and benchmark all ISA-eligible profiles."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
BENCH = ROOT / "python" / "demos" / "fft_bench.py"


def call(args, *, cwd=None, env=None, output=None, timeout=None, check=True):
    cmd = list(map(str, args))
    print("+", " ".join(cmd), flush=True)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout)
    else:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout)
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p


def get_cpu():
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    model = next((s.split(":", 1)[1].strip() for s in lines
                  if s.startswith("model name")), "unknown")
    flags = set(next((s.split(":", 1)[1].split() for s in lines
                      if s.startswith("flags")), []))
    cpus = sorted(os.sched_getaffinity(0))
    eligible = {"avx2", "fma", "f16c", "bmi1", "bmi2", "movbe"}
    v3 = eligible.issubset(flags)
    v4 = v3 and {"avx512f", "avx512cd", "avx512dq", "avx512bw", "avx512vl"}.issubset(flags)
    return {"model": model, "flags": sorted(flags), "cpu": cpus[0],
            "v3": v3, "v4": v4, "affinity": cpus}


def build_profile(profile, work, output):
    label = profile["label"]
    suffix = "".join(ch if ch.isalnum() else "-" for ch in label)
    source = work / "source" / suffix
    source.parent.mkdir(parents=True, exist_ok=True)
    call(["git", "worktree", "add", "--detach", str(source), profile["sha"]], cwd=REPO,
         output=output / "logs" / f"{suffix}-checkout.log", timeout=180)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source,
                                  text=True).strip()
    if sha != profile["sha"]:
        raise RuntimeError(f"{label}: source mismatch {sha} vs {profile['sha']}")
    expected = profile.get("compiler_defines", {})
    config = source / "fft_policy_profile.json"
    if expected:
        on_disk = json.loads(config.read_text())
        if expected != on_disk["compiler_defines"]:
            raise RuntimeError(f"{label}: branch compile defines changed")
    env = dict(os.environ, DUCC0_OPTIMIZATION="multiarch-strip",
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    if expected:
        env["DUCC0_CFLAGS"] = " ".join(f"-D{k}={v}" for k, v in sorted(expected.items()))
    builddir = work / "build" / suffix
    wheels = work / "wheels" / suffix
    wheels.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "wheel", "--no-cache-dir",
           "--no-deps", "--no-build-isolation",
           "-C", f"build-dir={builddir}", "-w", str(wheels), "."]
    call(cmd, cwd=source, env=env, output=output / "logs" / f"{suffix}-build.log", timeout=1800)
    found = list(wheels.glob("ducc0-*.whl"))
    if len(found) != 1:
        raise RuntimeError(f"{label}: expected 1 wheel, found {len(found)}")
    target = work / "installed" / suffix
    target.mkdir(parents=True, exist_ok=True)
    call([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
          "--target", str(target), str(found[0])], cwd=source, env=env,
         output=output / "logs" / f"{suffix}-install.log", timeout=600)
    return source, target, env


def env_for(base, target, isa):
    env = dict(base)
    env["PYTHONPATH"] = str(target)
    env["DUCC0_MAX_PSABI_LEVEL"] = str(isa)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--sample-ms", type=float, default=2.0)
    p.add_argument("--max-profiles", type=int, default=0,
                   help="0 means all pinned profiles")
    args = p.parse_args(argv)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    cpu = get_cpu()
    lock = json.loads((ROOT / "fft_policy_lock.json").read_text())
    status = {"cpu": cpu, "lock": lock, "profiles": [], "started": time.time()}
    (out / "status.json").write_text(json.dumps(status, indent=2))
    if not cpu["v3"]:
        status["skip_reason"] = "CPU does not support x86-64-v3 (AVX2/FMA/etc.)"
        status["finished"] = time.time()
        (out / "status.json").write_text(json.dumps(status, indent=2))
        print(status["skip_reason"])
        return 0
    allowed = (1, 2, 3, 4) if cpu["v4"] else (1, 2, 3)
    profiles = lock["profiles"][:args.max_profiles or None]
    for profile in profiles:
        label = profile["label"]
        row = {"profile": label, "branch": profile["branch"], "sha": profile["sha"],
               "defines": profile.get("compiler_defines", {}),
               "status": "pending", "isa": [], "started": time.time()}
        status["profiles"].append(row)
        (out / "status.json").write_text(json.dumps(status, indent=2))
        print(f"\n===== {label} ({profile['sha']}) =====", flush=True)
        try:
            source, target, buildenv = build_profile(profile, work, out)
            # The untouched DUCC reference is compiled once, then evaluated
            # using the exact x86-64-v1/v2/v3/v4 runtime multiarch selectors.
            isa_levels = allowed if label == "BASE" else ([3, 4] if cpu["v4"] else [3])
            for isa in isa_levels:
                name = f"{label}-v{isa}"
                phase = {"isa": isa, "status": "pending"}
                row["isa"].append(phase)
                env = env_for(buildenv, target, isa)
                try:
                    verify = ("import ducc0; actual=ducc0.misc.psabi_level(); "
                              f"assert actual == {isa}, (actual, {isa}); "
                              "print('DUCC selected psABI:',actual)")
                    call([sys.executable, "-c", verify], cwd=source, env=env,
                         output=out / "logs" / f"{name}-isa.log", timeout=180)
                    # Complete test suite for the source variant, not only a
                    # hand-picked FFT test. Limit each profile to avoid
                    # blocking remaining profiles if a test hangs.
                    if isa == (4 if cpu["v4"] else 3):
                        call([sys.executable, "-m", "pytest", "python/test", "-q", "-x"],
                             cwd=source, env=env, timeout=1200,
                             output=out / "logs" / f"{name}-tests.log")
                    dest = out / "runner_benchmarks" / name
                    dest.mkdir(parents=True, exist_ok=True)
                    cmd = ["taskset", "-c", str(cpu["cpu"]), sys.executable,
                           str(BENCH), "--policy-suite",
                           "--output", str(dest), "--profile", name,
                           "--samples", str(args.samples),
                           "--sample-ms", str(args.sample_ms), "--suite", "all"]
                    call(cmd, cwd=ROOT, env=env, timeout=7200,
                         output=out / "logs" / f"{name}-bench.log")
                    (dest / "source_manifest.json").write_text(json.dumps({
                        "branch": profile["branch"], "sha": profile["sha"],
                        "compiler_defines": profile.get("compiler_defines", {}),
                        "label": label, "isa": isa,
                        "ducc0_optimization": "multiarch-strip"}, indent=2))
                    # Save raw repeats but reduce artifact storage.
                    src = dest / "raw_repeats.csv"
                    with src.open("rb") as f, gzip.open(str(src) + ".gz", "wb",
                                                       compresslevel=6) as compressed:
                        shutil.copyfileobj(f, compressed)
                    src.unlink()
                    phase["status"] = "success"
                except Exception as e:
                    phase["status"] = "failed"
                    phase["error"] = f"{type(e).__name__}: {e}"
                    print(f"{name} FAILED: {e}", flush=True)
                finally:
                    (out / "status.json").write_text(json.dumps(status, indent=2))
            row["status"] = ("success" if any(x["status"] == "success" for x in row["isa"])
                             else "failed")
        except Exception as e:
            row["status"] = "build_failed"
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"{label} BUILD FAILED: {e}", flush=True)
        finally:
            row["finished"] = time.time()
            (out / "status.json").write_text(json.dumps(status, indent=2))
    status["finished"] = time.time()
    (out / "status.json").write_text(json.dumps(status, indent=2))
    successes = sum(p["status"] == "success" for p in status["profiles"])
    print(f"Completed profiles with benchmark results: {successes}/{len(profiles)}")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
