#!/usr/bin/env python3
"""Compare runtime-selected ARM64 DUCC FFT implementations on one native host.

Standalone CI benchmark, intentionally not part of the installed package.
The worker imports DUCC after the orchestrator sets DUCC0_MAX_ARM_PROFILE.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns

import numpy as np


# Include odd/even extents, both precisions, and c2c/r2c/c2r in 1D/2D/3D.
CASES = {
    "c2c-c16-1D-4095": ("c2c", "complex128", (4095,)),
    "c2c-c16-1D-4096": ("c2c", "complex128", (4096,)),
    "c2c-c16-2D-64x4095": ("c2c", "complex128", (64, 4095)),
    "c2c-c16-2D-64x4096": ("c2c", "complex128", (64, 4096)),
    "c2c-c8-1D-8192": ("c2c", "complex64", (8192,)),
    "c2c-c8-3D-32x32x32": ("c2c", "complex64", (32, 32, 32)),
    "r2c-f32-1D-8192": ("r2c", "float32", (8192,)),
    "r2c-f64-1D-8192": ("r2c", "float64", (8192,)),
    "r2c-f64-2D-256x256": ("r2c", "float64", (256, 256)),
    "c2r-f32-1D-8192": ("c2r", "float32", (8192,)),
    "c2r-f64-1D-8192": ("c2r", "float64", (8192,)),
    "c2r-f64-2D-256x256": ("c2r", "float64", (256, 256)),
}
PROFILES = ("neon", "sve", "sve2")


def pin_to_one_cpu():
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            try:
                os.sched_setaffinity(0, {allowed[0]})
                return allowed[0]
            except OSError as exc:
                print("CPU affinity unavailable:", exc, file=sys.stderr)
    return None


def worker(args):
    # Import only after the selector environment has been set.
    import ducc0

    info = ducc0.misc.cpu_info()
    if info["architecture"] != "aarch64" or not info["multiarch"]:
        raise RuntimeError("benchmark requires a Linux ARM64 multibuild")
    if info["compiled_profiles"] != list(PROFILES):
        raise RuntimeError("all three ARM64 variants must be compiled: " + repr(info))
    if info["active_profile"] != args.profile:
        raise RuntimeError("wrong active profile: " + repr(info))
    if os.environ.get("DUCC0_MAX_ARM_PROFILE") != args.profile:
        raise RuntimeError("profile override missing or inconsistent")

    cpu = pin_to_one_cpu()
    operation, dtype_name, shape = CASES[args.case]
    dtype = np.dtype(dtype_name)
    seed = 42 + sum(map(ord, args.case)) + args.sample * 1009
    rng = np.random.default_rng(seed)
    real = (rng.random(shape) - 0.5).astype(
        np.float32 if dtype in (np.dtype("float32"), np.dtype("complex64"))
        else np.float64)
    axes = tuple(range(len(shape)))

    if operation == "c2c":
        data = (real + 1j * (rng.random(shape) - 0.5)).astype(dtype)
        out = np.empty(shape, dtype=dtype)
        expected = np.fft.fftn(data, axes=axes)
        def transform():
            return ducc0.fft.c2c(data, axes=axes, forward=True,
                                 inorm=0, out=out, nthreads=1)
    elif operation == "r2c":
        data = real
        cdtype = np.complex64 if dtype == np.dtype("float32") else np.complex128
        out = np.empty(shape[:-1] + (shape[-1] // 2 + 1,), dtype=cdtype)
        expected = np.fft.rfftn(data, axes=axes)
        def transform():
            return ducc0.fft.r2c(data, axes=axes, forward=True,
                                 inorm=0, out=out, nthreads=1)
    else:
        cdtype = np.complex64 if dtype == np.dtype("float32") else np.complex128
        data = np.fft.rfftn(real, axes=axes).astype(cdtype)
        out = np.empty(shape, dtype=dtype)
        expected = np.fft.irfftn(data, s=shape, axes=axes, norm="forward")
        def transform():
            return ducc0.fft.c2r(data, axes=axes, lastsize=shape[-1],
                                 forward=False, inorm=0, out=out, nthreads=1)

    actual = transform()
    if actual.shape != expected.shape:
        raise RuntimeError("FFT output shape mismatch")
    denom = max(float(np.linalg.norm(expected.ravel())), 1e-30)
    rel_l2 = float(np.linalg.norm((actual - expected).ravel()) / denom)
    limit = 2e-5 if dtype in (np.dtype("complex64"), np.dtype("float32")) else 1e-10
    if not np.isfinite(rel_l2) or rel_l2 > limit:
        raise RuntimeError(
            f"FFT mismatch: {args.case} {args.profile} relative L2 {rel_l2} > {limit}"
        )

    for _ in range(args.warmup):
        transform()
    timings = []
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for _ in range(args.repeat):
            start = perf_counter_ns()
            transform()
            timings.append((perf_counter_ns() - start) / 1e6)
    finally:
        if was_enabled:
            gc.enable()

    print(json.dumps({
        "case": args.case, "profile": args.profile, "sample": args.sample,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings), "max_ms": max(timings),
        "relative_l2": rel_l2, "cpu_affinity": cpu,
        "shape": list(shape), "dtype": dtype_name, "operation": operation,
        "cpu_info": info,
    }, sort_keys=True))


def orchestrate(args):
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    import ducc0

    info = ducc0.misc.cpu_info()
    if info.get("architecture") != "aarch64" or not info.get("multiarch"):
        raise RuntimeError("requires an ARM64 multiarch build")
    if info["compiled_profiles"] != list(PROFILES) or        info["available_profiles"] != list(PROFILES):
        raise RuntimeError("NEON, SVE and SVE2 are all required: " + repr(info))
    (result_dir / "cpu-info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n")

    records = []
    script = str(Path(__file__).resolve())
    with (result_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for sample in range(args.samples):
            for case_index, case in enumerate(CASES):
                # Rotate order to avoid giving any one profile a fixed advantage.
                offset = (sample + case_index) % len(PROFILES)
                for i in range(len(PROFILES)):
                    profile = PROFILES[(offset + i) % len(PROFILES)]
                    env = os.environ.copy()
                    env.update({
                        "DUCC0_MAX_ARM_PROFILE": profile,
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    })
                    command = [
                        sys.executable, script, "--worker", "--case", case,
                        "--profile", profile, "--sample", str(sample),
                        "--repeat", str(args.repeat),
                        "--warmup", str(args.warmup),
                    ]
                    completed = subprocess.run(command, env=env,
                                               check=True, capture_output=True,
                                               text=True)
                    record = json.loads(completed.stdout.strip())
                    records.append(record)
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                    f.flush()
                    print(f"{case} sample={sample} {profile}: "
                          f"{record['median_ms']:.3f} ms, "
                          f"L2={record['relative_l2']:.2g}", flush=True)

    lines = [
        "# ARM64 DUCC FFT dispatch benchmark",
        "",
        "Same native ARM64 runner and wheel; each profile selected in a fresh "
        "process at import. Single-thread FFTs, warmups and repeated measurements.",
        f"{args.samples} independent input samples, {args.repeat} timed repetitions "
        "per sample. Order rotated across profiles.",
        "",
        "| FFT case | NEON ms | SVE ms | SVE2 ms | SVE/NEON | SVE2/NEON |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        times = {
            p: statistics.median(r["median_ms"] for r in records
                                 if r["case"] == case and r["profile"] == p)
            for p in PROFILES
        }
        lines.append(
            f"| {case} | {times['neon']:.3f} | {times['sve']:.3f} | "
            f"{times['sve2']:.3f} | {times['neon']/times['sve']:.2f}x | "
            f"{times['neon']/times['sve2']:.2f}x |"
        )
    lines += [
        "", "Ratios >1 favor the named profile. Hosted runners are noisy; "
        "these numbers are indicative rather than stable performance guarantees. "
        "A selected SVE/SVE2 profile does not demonstrate SVE2-specific instructions "
        "or a full-width SVE native SIMD backend.",
        "",
    ]
    report = "\n".join(lines)
    (result_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-dir", default="arm64-fft-bench")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.samples < 1:
        parser.error("invalid sample/repeat/warmup count")
    if args.worker:
        if not args.case or not args.profile:
            parser.error("--worker requires --case and --profile")
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
