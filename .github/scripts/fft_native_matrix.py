#!/usr/bin/env python3
"""Eight single-ISA native comparisons on the same CPU as the multiarch matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from fft_factorial_benchmark import (
    CASES, _cpu_identity, measure_reference_sample,
)
from fft_factorial_run import (
    ROOT, SCRIPT_DIR, BENCHMARK_SCRIPT, _has_lto, append_jsonl, capture,
    clean_build_env, command_tokens, compiler_metadata, find_compile_commands,
    find_final_link_command, run_logged, stop_worker, write_json,
)
from prepare_factorial_sources import (
    BASE_SHA, CONFIGS, ensure_base_commit, prepare_variant,
)


def native_configs(profile: str) -> list[dict]:
    return [{**base, "id": f"N{base['id']}{int(lto)}",
             "source_variant": base["id"], "native_profile": profile,
             "native_lto": lto, "build_kind": "native"}
            for lto in (False, True) for base in CONFIGS]


def compile_native(config: dict, source: Path, artifact: Path,
                   build_root: Path, compiler: str, jobs: int) -> dict:
    variant = config["id"]
    build_dir = build_root / "build" / variant
    install_dir = build_root / "install" / variant
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    log = artifact / "native-build-logs" / f"{variant}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")
    env = clean_build_env(compiler)
    env["DUCC0_OPTIMIZATION"] = "native"
    version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    march = config["native_profile"]
    command = [
        "cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DSKBUILD_PROJECT_NAME=ducc0", f"-DSKBUILD_PROJECT_VERSION={version}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DDUCC0_ARCH_FLAGS=-march={march}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    timing = {}
    for label, cmd in (("configure", command),
                       ("build", ["cmake", "--build", str(build_dir),
                                  "--parallel", str(jobs), "--verbose"]),
                       ("install", ["cmake", "--install", str(build_dir)])):
        rc, elapsed, output = run_logged(cmd, log, label, source, env)
        timing[label + "_seconds"] = elapsed
        if rc:
            raise RuntimeError(f"{variant} {label} failed: {output[-4500:]}")
    extensions = list(install_dir.glob("ducc0*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"{variant}: expected one extension: {extensions}")
    extension = extensions[0]
    compile_commands = find_compile_commands(build_dir)
    relevant = [entry for entry in compile_commands
                if "ducc0_lib.dir" in " ".join(command_tokens(entry))
                or "CMakeFiles/ducc0.dir" in " ".join(command_tokens(entry))]
    if not relevant:
        raise RuntimeError(f"{variant}: no expected native compile commands")
    for entry in relevant:
        tokens = command_tokens(entry)
        if f"-march={march}" not in tokens:
            raise RuntimeError(f"{variant}: wrong ISA compile target: {entry}")
        if _has_lto(tokens) != config["native_lto"]:
            raise RuntimeError(f"{variant}: wrong LTO compile state: {entry}")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    link = find_final_link_command(log_text)
    if _has_lto(shlex.split(link)) != config["native_lto"]:
        raise RuntimeError(f"{variant}: wrong final LTO link state: {link}")
    sections = capture(["readelf", "-S", str(extension)])
    if sections.returncode or ".gnu.lto" in sections.stdout:
        raise RuntimeError(f"{variant}: final extension has LTO IR or readelf failed")
    # Query the module in its own Python process: no imports from incompatible ISAs.
    env["PYTHONPATH"] = str(install_dir) + os.pathsep + env.get("PYTHONPATH", "")
    probe = capture([sys.executable, "-c",
                     "import ducc0,json; print(json.dumps(ducc0.misc.cpu_info()))"],
                    env=env, timeout=120)
    if probe.returncode:
        raise RuntimeError(f"{variant}: import failed: {probe.stdout[-2500:]}")
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    if info.get("multiarch") is not False or info.get("architecture") != "x86-64":
        raise RuntimeError(f"{variant}: expected single-ISA native extension: {info}")
    return {"build_directory": str(build_dir), "install_directory": str(install_dir),
            "extension": str(extension), "size_bytes": extension.stat().st_size,
            "build_times": timing, "compile_command_count": len(relevant),
            "march": march, "lto": config["native_lto"],
            "final_link": link, "cpu_info": info}


def launch(config: dict, diagnostics: dict, profile: str, artifact: Path, nrepeat: int):
    env = os.environ.copy()
    env.pop("DUCC0_MAX_PSABI_LEVEL", None)
    env["PYTHONPATH"] = (diagnostics["install_directory"] + os.pathsep +
                         env.get("PYTHONPATH", ""))
    log_path = artifact / "native-worker-logs" / f"{config['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    cmd = [sys.executable, str(BENCHMARK_SCRIPT), "--worker", "--native",
           "--variant", config["id"], "--profile", profile,
           "--nrepeat", str(nrepeat)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=log, text=True, bufsize=1, env=env)
    first = process.stdout.readline()
    if not first:
        rc = process.wait()
        log.close()
        raise RuntimeError(f"{config['id']} worker failed to import, return code {rc}; {log_path}")
    ready = json.loads(first)
    if (ready.get("record_type") != "worker_ready"
            or ready.get("cpu_info", {}).get("active_profile") != profile
            or ready.get("cpu_info", {}).get("multiarch") is not False):
        process.kill()
        process.wait()
        log.close()
        raise RuntimeError(f"{config['id']} unexpected native worker info: {ready}")
    return {"process": process, "log": log, "log_path": str(log_path)}


def run_native(artifact: Path, configs: list[dict], built: dict,
               profile: str, ntry: int, nrepeat: int, expected_dir: Path) -> dict:
    import statistics

    timing_path = artifact / "native-timings.jsonl"
    reference_path = artifact / "native-references.jsonl"
    timing_path.write_text("", encoding="utf-8")
    reference_path.write_text("", encoding="utf-8")
    workers = {}
    errors = []
    for config in configs:
        if config["id"] not in built:
            continue
        try:
            workers[config["id"]] = launch(config, built[config["id"]],
                                            profile, artifact, nrepeat)
        except Exception as exc:
            errors.append({"variant": config["id"], "error": f"{type(exc).__name__}: {exc}"})
    try:
        for case_index, case in enumerate(CASES):
            for sample in range(ntry):
                expected_path = expected_dir / profile / case["id"] / f"{sample}.npy"
                rows, fingerprint = measure_reference_sample(
                    case, sample, profile, nrepeat, expected_path)
                for row in rows:
                    row["measurement_group"] = "native"
                    append_jsonl(reference_path, row)
                    if row.get("correctness") != "pass":
                        errors.append({"reference": row.get("reference"), "case": case["id"],
                                       "sample": sample, "error": row.get("correctness_error")})
                witness = any(row["reference"] == "numpy" and
                              row["correctness"] == "pass" for row in rows)
                rotation = (case_index + sample) % len(configs)
                order = configs[rotation:] + configs[:rotation]
                for config in order:
                    variant = config["id"]
                    worker = workers.get(variant)
                    if worker is None:
                        continue
                    process = worker["process"]
                    if process.poll() is not None:
                        errors.append({"variant": variant, "case": case["id"],
                                       "error": f"worker ended: {process.returncode}"})
                        continue
                    cmd = {"case": case["id"], "sample_index": sample,
                           "input_sha256": fingerprint,
                           "expected_path": str(expected_path) if witness else None}
                    try:
                        process.stdin.write(json.dumps(cmd) + "\n")
                        process.stdin.flush()
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError("worker produced no timing result")
                        record = json.loads(line)
                        if (record.get("record_type") != "ducc"
                                or record.get("variant") != variant
                                or record.get("case") != case["id"]
                                or record.get("sample_index") != sample):
                            raise RuntimeError(f"unexpected worker output: {record}")
                        record["measurement_group"] = "native"
                        record["native_lto"] = config["native_lto"]
                        record["native_isa"] = profile
                        record["source_variant"] = config["source_variant"]
                        append_jsonl(timing_path, record)
                        if record.get("correctness") != "pass":
                            errors.append({"variant": variant, "case": case["id"],
                                           "sample": sample, "error": record.get("correctness_error")})
                    except Exception as exc:
                        errors.append({"variant": variant, "case": case["id"],
                                       "sample": sample, "error": str(exc)})
                        stop_worker(worker)
                        del workers[variant]
            print(f"native {profile} complete {case['id']}", flush=True)
    finally:
        for worker in workers.values():
            stop_worker(worker)
    data = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    expected = len(configs) * len(CASES) * ntry
    return {"expected_samples": expected, "completed_samples": len(data),
            "correctness_failures": sum(row["correctness"] != "pass" for row in data),
            "errors": errors, "status": "pass" if len(data) == expected and not errors else "fail"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--native-profile", choices=("auto", "v3", "v4"), default="auto")
    parser.add_argument("--ntry", type=int, default=3)
    parser.add_argument("--nrepeat", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    artifact = args.output_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    multiarch = json.loads((artifact / "run-status.json").read_text())
    available = multiarch.get("supported_profiles", [])
    selection = {"v3": "x86-64-v3", "v4": "x86-64-v4"}
    profile = (next((p for p in ("x86-64-v4", "x86-64-v3") if p in available), None)
               if args.native_profile == "auto" else selection[args.native_profile])
    identity = _cpu_identity()
    status = {"requested_native_profile": args.native_profile,
              "selected_native_profile": profile, "available_profiles": available,
              "cpu_identity": identity, "host_multiarch_cpu_info": multiarch.get("cpu_info", {}),
              "host_os": platform.platform(), "native_configuration_count": 8,
              "variants": {}, "status": "pending"}
    status_path = artifact / "native-status.json"
    write_json(status_path, status)
    print("Host CPU:", identity.get("model"), flush=True)
    print("Available ISA profiles:", ", ".join(available), flush=True)
    print("Native selection:", profile, flush=True)
    if profile not in available or profile not in selection.values():
        status["status"] = "fail" if args.native_profile != "auto" else "skipped"
        status["error"] = f"target ISA {profile} unsupported by this runner"
        write_json(status_path, status)
        return 1 if status["status"] == "fail" else 0

    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        raise RuntimeError("C++ compiler is unavailable")
    compiler = str(Path(compiler).resolve())
    status["compiler"] = compiler_metadata(compiler)
    configs = native_configs(profile)
    worktrees = args.worktree_root.resolve()
    worktrees.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="ducc-native-build-", dir=worktrees.parent))
    started = time.perf_counter()
    built = {}
    try:
        ensure_base_commit(ROOT)
        for config in configs:
            variant = config["id"]
            item = {"lto": config["native_lto"], "isa": profile,
                    "inline": config["special_mul_fix"], "tweaks": config["fft_tweaks"]}
            status["variants"][variant] = item
            source = worktrees / variant
            try:
                subprocess.run(["git", "worktree", "add", "--detach", str(source), BASE_SHA],
                               cwd=ROOT, check=True, capture_output=True, text=True)
                item["provenance"] = prepare_variant(source, config, ROOT)
                diagnostics = compile_native(config, source, artifact, build_root,
                                             compiler, args.jobs)
                built[variant] = diagnostics
                item["build"] = "pass"
                item["build_diagnostics"] = diagnostics
            except Exception as exc:
                item["build"] = "fail"
                item["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                write_json(status_path, status)
            print(f"native build {variant}: {item['build']}", flush=True)
        status["benchmark"] = run_native(artifact, configs, built, profile,
                                         args.ntry, args.nrepeat,
                                         worktrees / "native-expected")
        status["elapsed_seconds"] = time.perf_counter() - started
        status["status"] = ("pass" if all(v.get("build") == "pass"
                            for v in status["variants"].values()) and
                            status["benchmark"]["status"] == "pass" else "fail")
        write_json(status_path, status)
        return 0 if status["status"] == "pass" else 1
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
        for config in configs:
            subprocess.run(["git", "worktree", "remove", "--force",
                            str(worktrees / config["id"])], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        shutil.rmtree(worktrees / "native-expected", ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
