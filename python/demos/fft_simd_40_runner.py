#!/usr/bin/env python3
"""CPU-paired scientific FFT benchmark: simplified SIMD branch vs upstream.

Each of forty runners builds the same pinned upstream and candidate sources.
Every runner executes eight scientific FFT families under all supported
x86-64 v1/v3/v4 profiles with eleven timing samples per case.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = "757145c328d4680fc23bf37e372860fa49c4d8b5"
VERSIONS = {
    "upstream": BASE,
    "simd": "f80b987891bd34d367c414dc2f27a13e42f4cb94",
}
SUITES = ("1d", "batched", "strided", "nd", "long", "real",
          "r2r", "convolution")
PROFILES = (
    ("upstream", 1), ("upstream", 3), ("upstream", 4),
    ("simd", 3), ("simd", 4),
)

def run(cmd, *, env=None, log=None, check=True):
    start = time.monotonic()
    printable = " ".join(map(str, cmd))
    print(f"$ {printable}", flush=True)
    if log is None:
        cp = subprocess.run(cmd, env=env, text=True, check=False)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            f.write(f"command: {printable}\n")
            f.flush()
            cp = subprocess.run(cmd, env=env, stdout=f,
                                stderr=subprocess.STDOUT,
                                text=True, check=False)
    duration = time.monotonic() - start
    print(f"exit={cp.returncode} seconds={duration:.1f} {printable}",
          flush=True)
    if check and cp.returncode:
        raise RuntimeError(f"command failed, exit={cp.returncode}: "
                           f"{printable}, log={log}")
    return cp.returncode


def env_for(install, variant, level, root):
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(install) + os.pathsep +
                         os.environ.get("PYTHONPATH", ""))
    env["DUCC0_MAX_PSABI_LEVEL"] = str(level)
    env["DUCC_EXPECTED_INSTALL"] = str(install)
    env["DUCC_SOURCE_SHA"] = VERSIONS[variant]
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_module(env, log):
    script = """
import ducc0, os
from pathlib import Path
want = int(os.environ['DUCC0_MAX_PSABI_LEVEL'])
actual = int(ducc0.misc.psabi_level())
path = Path(ducc0.__file__).resolve()
assert actual == want, (actual, want)
assert path.is_relative_to(Path(os.environ['DUCC_EXPECTED_INSTALL']).resolve()), path
print('active PSABI', actual, 'module', path, 'wrapper', ducc0.__wrapper__)
"""
    return run([sys.executable, "-c", script], env=env,
               log=log, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not 1 <= args.replicate <= 40:
        parser.error("replicate must be 1..40")
    repo = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    root = Path(os.environ["RUNNER_TEMP"]).resolve() / "ducc-ablation"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_log_dir = output / "build-logs"
    build_log_dir.mkdir(exist_ok=True)
    (output / "versions.json").write_text(json.dumps(
        dict(replicate=args.replicate, versions=VERSIONS,
             suites=SUITES, profiles=PROFILES,
             optimization=os.environ.get("DUCC0_OPTIMIZATION"),
             note="Original upstream and simplified SIMD sources are "
                  "built independently with cache enabled and identical "
                  "multiarch-strip flags; each runner executes identical "
                  "scientific workload cases."),
        indent=2) + "\n")
    fail = []
    for variant, sha in VERSIONS.items():
        src = root / "sources" / variant
        dist = root / "wheels" / variant
        installed = root / "installed" / variant
        for d in (src.parent, dist, installed, root / "builds"):
            d.mkdir(parents=True, exist_ok=True)
        run(["git", "cat-file", "-e", f"{sha}^{{commit}}"])
        run(["git", "worktree", "add", "--detach", str(src), sha],
            log=build_log_dir / f"{variant}-checkout.txt")
        build_env = os.environ.copy()
        # Match compiler options across the three independently built sources.
        build_env.pop("CXXFLAGS", None)
        run([sys.executable, "-m", "pip", "wheel",
             "--no-build-isolation", "--no-cache-dir", "--no-deps",
             "-C", f"build-dir={root / 'builds' / variant}",
             "-w", str(dist), str(src)],
            env=build_env, log=build_log_dir / f"{variant}-build.txt")
        wheels = sorted(dist.glob("ducc0-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel for {variant}: {wheels}")
        run([sys.executable, "-m", "pip", "install", "--no-deps",
             "--no-cache-dir", "--target", str(installed),
             str(wheels[0])],
            log=build_log_dir / f"{variant}-install.txt")
    max_levels = {}
    for variant in VERSIONS:
        env = env_for(root / "installed" / variant, variant, 3, root)
        env.pop("DUCC0_MAX_PSABI_LEVEL", None)
        p = subprocess.run(
            [sys.executable, "-c",
             "import ducc0; print(ducc0.misc.psabi_level())"],
            env=env, capture_output=True, text=True, check=True)
        max_levels[variant] = int(p.stdout.strip())
        if max_levels[variant] < 3:
            raise RuntimeError(f"v3 unavailable for {variant}")
    usable = min(max_levels.values())
    (output / "hardware.json").write_text(json.dumps(
        dict(runner=os.environ.get("RUNNER_NAME"),
             runner_os=os.environ.get("RUNNER_OS"),
             levels=max_levels, usable_level=usable,
             replicate=args.replicate), indent=2) + "\n")
    # Confirm mathematical FFT parity once per independently built profile.
    for variant, level in PROFILES:
        if level > usable:
            continue
        env = env_for(root / "installed" / variant, variant, level, root)
        log = output / f"correctness-{variant}-v{level}.txt"
        result = run([sys.executable, "-m", "pytest",
                      str(root / "sources" / variant /
                          "python/test/test_fft.py"), "-q", "-x"],
                     env=env, log=log, check=False)
        if result:
            fail.append(f"correctness {variant}-v{level}")
    # Execute ALL FFT suites and source/ISA combinations on the SAME runner.
    for suite_idx, suite in enumerate(SUITES):
        offset = (args.replicate + suite_idx) % len(PROFILES)
        rotated = PROFILES[offset:] + PROFILES[:offset]
        for variant, level in rotated:
            directory = output / suite / f"{variant}-v{level}"
            directory.mkdir(parents=True, exist_ok=True)
            if level > usable:
                (directory / "skip.txt").write_text(
                    f"PSABI v{level} unavailable; maximum v{usable}\n")
                continue
            env = env_for(root / "installed" / variant, variant, level, root)
            profile = f"{variant}-v{level}"
            if check_module(env, directory / "import.txt"):
                fail.append(f"import {suite} {profile}")
                continue
            rc = run([sys.executable,
                      str(repo / "python/demos/fft_scientific_40_matrix.py"),
                      "--suite", suite,
                      "--profile", profile,
                      "--shard", str(args.replicate - 1),
                      "--replicate", str(args.replicate),
                      "--scientific",
                      "--samples", "11", "--target-ms", "4",
                      "--output", str(directory)],
                     env=env, log=directory / "benchmark.txt",
                     check=False)
            if rc:
                fail.append(f"benchmark {suite} {profile}")
    (output / "errors.json").write_text(json.dumps(
        dict(failures=fail, count=len(fail)), indent=2) + "\n")
    if fail:
        raise SystemExit(f"{len(fail)} failed commands: " +
                         ", ".join(fail[:15]))
    print("Completed eight scientific FFT suites across two pinned "
          "source builds and 11 timing samples per case.",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
