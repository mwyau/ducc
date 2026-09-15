#!/usr/bin/env python3
"""Deterministic public-API SIMD correctness and timing checks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


SEED = 20260914


def nalm(lmax: int, mmax: int) -> int:
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def measure(execute, output, warmups: int, repetitions: int) -> tuple[dict, np.ndarray]:
    for _ in range(warmups):
        execute()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        execute()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    snapshot = np.array(output, copy=True)
    stats = {
        "median_ms": float(statistics.median(samples)),
        "minimum_ms": float(min(samples)),
        "maximum_ms": float(max(samples)),
        "stdev_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "checksum": float(np.sum(snapshot.real, dtype=np.float64)
                          + np.sum(snapshot.imag, dtype=np.float64)),
        "samples_ms": samples,
    }
    return stats, snapshot


def run_case(case: str, output_dir: Path, warmups: int, repetitions: int,
             threads: int) -> None:
    import ducc0

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}

    if case == "nufft":
        npoints = 32768
        coord = (rng.random((npoints, 2)).astype(np.float32)*np.float32(2*np.pi)
                 - np.float32(np.pi)).astype(np.float32)
        grid = (rng.random((128, 128)).astype(np.float32) - np.float32(.5)
                + 1j*(rng.random((128, 128)).astype(np.float32)-np.float32(.5)))
        grid = grid.astype(np.complex64)
        output = np.empty(npoints, dtype=np.complex64)
        plan = ducc0.nufft.plan(nu2u=False, coord=coord, grid_shape=grid.shape,
                                epsilon=3e-5, nthreads=threads)

        def execute():
            plan.u2nu(grid=grid, forward=True, out=output)

    elif case == "sht":
        lmax = 160
        mmax = 160
        alm = (rng.standard_normal((1, nalm(lmax, mmax)))
               + 1j*rng.standard_normal((1, nalm(lmax, mmax)))).astype(np.complex128)
        output = np.empty((1, lmax+2, 2*mmax+2), dtype=np.float64)

        def execute():
            ducc0.sht.synthesis_2d(
                alm=alm, spin=0, lmax=lmax, mmax=mmax, geometry="CC",
                ntheta=output.shape[1], nphi=output.shape[2], nthreads=threads,
                map=output)

    else:
        raise ValueError(f"unknown case: {case}")

    stats, snapshot = measure(execute, output, warmups, repetitions)
    np.save(output_dir/(case+".npy"), snapshot)
    results[case] = stats
    (output_dir/"results.json").write_text(json.dumps({
        "case": case, "seed": SEED, "threads": threads,
        "warmups": warmups, "repetitions": repetitions,
        "module": str(Path(ducc0.__file__).resolve()), "results": results,
    }, indent=2)+"\n", encoding="utf-8")
    print(f"{case}: median_ms={stats['median_ms']:.6f} "
          f"minimum_ms={stats['minimum_ms']:.6f} "
          f"stdev_ms={stats['stdev_ms']:.6f} "
          f"checksum={stats['checksum']:.12g}", flush=True)


def aggregate(values: list[float]) -> tuple[float, float, float, float]:
    return (float(statistics.median(values)), float(min(values)),
            float(max(values)),
            float(statistics.stdev(values)) if len(values) > 1 else 0.0)


def compare(case: str, baseline_dirs: list[Path], candidate_dirs: list[Path],
            summary_file: Path | None, require_improvement: bool = False) -> None:
    if len(baseline_dirs) != len(candidate_dirs) or not baseline_dirs:
        raise SystemExit("baseline and candidate run counts must match and be nonzero")
    base = np.load(baseline_dirs[0]/(case+".npy"))
    cand = np.load(candidate_dirs[0]/(case+".npy"))
    if base.shape != cand.shape or base.dtype != cand.dtype:
        raise SystemExit(f"{case}: shape/dtype mismatch {base.shape}/{base.dtype} "
                         f"vs {cand.shape}/{cand.dtype}")
    diff = np.abs(cand-base)
    real_dtype = np.empty((), dtype=base.dtype).real.dtype
    denominator = np.maximum(np.abs(base), np.finfo(real_dtype).tiny)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff/denominator))
    if case == "sht":
        rtol = atol = 1e-10
    else:
        rtol = atol = 3e-5
    accuracy_ok = bool(np.allclose(cand, base, rtol=rtol, atol=atol))

    base_stats = [json.loads((d/"results.json").read_text())["results"][case]
                  for d in baseline_dirs]
    cand_stats = [json.loads((d/"results.json").read_text())["results"][case]
                  for d in candidate_dirs]
    paired_speedups = [b["median_ms"]/c["median_ms"]
                       for b, c in zip(base_stats, cand_stats)]
    paired_wins = sum(speedup >= 1.01 for speedup in paired_speedups)
    base_timing = aggregate([x["median_ms"] for x in base_stats])
    cand_timing = aggregate([x["median_ms"] for x in cand_stats])
    speedup = base_timing[0]/cand_timing[0]
    improvement = (speedup-1.)*100.
    repeatable = paired_wins > len(paired_speedups)//2
    if improvement > 1.0 and repeatable:
        performance = "verified improvement"
    elif improvement > 1.0:
        performance = "inconsistent improvement"
    elif improvement < -1.0:
        performance = "regression"
    else:
        performance = "no measurable difference"

    report = (
        f"| {case} | {max_abs:.9g} | {max_rel:.9g} | "
        f"{'PASS' if accuracy_ok else 'FAIL'} | "
        f"{base_timing[0]:.6f} | {base_timing[1]:.6f} | "
        f"{base_timing[2]-base_timing[1]:.6f} | "
        f"{cand_timing[0]:.6f} | {cand_timing[1]:.6f} | "
        f"{cand_timing[2]-cand_timing[1]:.6f} | "
        f"{speedup:.4f}x | {improvement:+.2f}% | "
        f"{paired_wins}/{len(paired_speedups)} | {performance} |\n"
    )
    print(report, end="")
    print(f"{case}: accuracy max_abs={max_abs:.9g} max_rel={max_rel:.9g} "
          f"tolerance={rtol:g} accuracy={'PASS' if accuracy_ok else 'FAIL'}")
    print(f"{case}: baseline median_ms={base_timing[0]:.6f} "
          f"best_ms={base_timing[1]:.6f} spread_ms={base_timing[2]-base_timing[1]:.6f} "
          f"stdev_ms={base_timing[3]:.6f}")
    print(f"{case}: candidate median_ms={cand_timing[0]:.6f} "
          f"best_ms={cand_timing[1]:.6f} spread_ms={cand_timing[2]-cand_timing[1]:.6f} "
          f"stdev_ms={cand_timing[3]:.6f} "
          f"ratio={speedup:.4f} improvement={improvement:+.2f}% "
          f"paired_faster_by_1pct={paired_wins}/{len(paired_speedups)} status={performance}")
    if summary_file is not None:
        summary_file.write_text(
            "| Case | Max abs error | Max rel error | Accuracy | "
            "Baseline median (ms) | Baseline best (ms) | Baseline spread (ms) | "
            "Candidate median (ms) | Candidate best (ms) | Candidate spread (ms) | "
            "Ratio | Percent change | Paired >=1% faster | Status |\n"
            "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
            + report, encoding="utf-8")
    if not accuracy_ok:
        raise SystemExit(f"{case}: focused accuracy comparison failed")
    if require_improvement and performance != "verified improvement":
        raise SystemExit(f"{case}: no repeatable improvement was measured "
                         f"(paired >=1% faster {paired_wins}/{len(paired_speedups)})")


def compare_hsum(baseline_dirs: list[Path], candidate_dirs: list[Path],
                 summary_file: Path | None, label: str) -> None:
    if len(baseline_dirs) != len(candidate_dirs) or not baseline_dirs:
        raise SystemExit("baseline and candidate run counts must match and be nonzero")
    base = np.fromfile(baseline_dirs[0]/"values.bin", dtype=np.float32)
    cand = np.fromfile(candidate_dirs[0]/"values.bin", dtype=np.float32)
    if base.shape != cand.shape:
        raise SystemExit("SSE hsum output length mismatch")
    diff = np.abs(cand-base)
    max_abs = float(np.max(diff))
    denominator = np.maximum(np.abs(base), np.finfo(np.float32).tiny)
    max_rel = float(np.max(diff/denominator))
    accuracy_ok = bool(np.allclose(cand, base, rtol=3e-5, atol=3e-5))
    base_stats = [json.loads((d/"stats.json").read_text()) for d in baseline_dirs]
    cand_stats = [json.loads((d/"stats.json").read_text()) for d in candidate_dirs]
    paired_speedups = [b["median_ns"]/c["median_ns"]
                       for b, c in zip(base_stats, cand_stats)]
    paired_wins = sum(speedup > 1.0 for speedup in paired_speedups)
    b = aggregate([x["median_ns"] for x in base_stats])
    c = aggregate([x["median_ns"] for x in cand_stats])
    speedup = b[0]/c[0]
    improvement = (speedup-1.)*100.
    status = ("verified improvement" if improvement > 1.0 and
              paired_wins > len(paired_speedups)//2 else
              "inconsistent improvement" if improvement > 1.0 else
              "regression" if improvement < -1.0 else
              "no measurable difference")
    report = (
        f"| {label} | {max_abs:.9g} | {max_rel:.9g} | "
        f"{'PASS' if accuracy_ok else 'FAIL'} | {b[0]:.4f} | {b[1]:.4f} | "
        f"{b[2]-b[1]:.4f} | {c[0]:.4f} | {c[1]:.4f} | "
        f"{c[2]-c[1]:.4f} | {speedup:.4f}x | {improvement:+.2f}% | "
        f"{paired_wins}/{len(paired_speedups)} | {status} |\n")
    print(report, end="")
    print(f"{label}: accuracy max_abs={max_abs:.9g} max_rel={max_rel:.9g} "
          f"accuracy={'PASS' if accuracy_ok else 'FAIL'}")
    print(f"{label}: baseline median_ns={b[0]:.4f} best_ns={b[1]:.4f} "
          f"spread_ns={b[2]-b[1]:.4f} stdev_ns={b[3]:.4f}")
    print(f"{label}: candidate median_ns={c[0]:.4f} best_ns={c[1]:.4f} "
          f"spread_ns={c[2]-c[1]:.4f} stdev_ns={c[3]:.4f} "
          f"ratio={speedup:.4f} improvement={improvement:+.2f}% "
          f"paired_faster={paired_wins}/{len(paired_speedups)} status={status}")
    if summary_file is not None:
        summary_file.write_text(
            "| Target | Max abs error | Max rel error | Accuracy | "
            "Baseline median (ns) | Baseline best (ns) | Baseline spread (ns) | "
            "Candidate median (ns) | Candidate best (ns) | Candidate spread (ns) | "
            "Ratio | Percent change | Paired faster | Status |\n"
            "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
            + report, encoding="utf-8")
    if not accuracy_ok:
        raise SystemExit(f"{label} focused accuracy comparison failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", choices=("nufft", "sht"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--warmups", type=int, default=2)
    run.add_argument("--repetitions", type=int, default=6)
    run.add_argument("--threads", type=int, default=1)
    comp = sub.add_parser("compare")
    comp.add_argument("--case", choices=("nufft", "sht"), required=True)
    comp.add_argument("--baseline", type=Path, action="append", required=True)
    comp.add_argument("--candidate", type=Path, action="append", required=True)
    comp.add_argument("--summary", type=Path)
    comp.add_argument("--require-improvement", action="store_true")
    hsum = sub.add_parser("compare-hsum")
    hsum.add_argument("--baseline", type=Path, action="append", required=True)
    hsum.add_argument("--candidate", type=Path, action="append", required=True)
    hsum.add_argument("--summary", type=Path)
    hsum.add_argument("--label", default="SSE2-only/no-AVX")
    args = parser.parse_args()
    if args.command == "run":
        run_case(args.case, args.output, args.warmups, args.repetitions, args.threads)
    elif args.command == "compare":
        compare(args.case, args.baseline, args.candidate, args.summary,
                args.require_improvement)
    else:
        compare_hsum(args.baseline, args.candidate, args.summary, args.label)


if __name__ == "__main__":
    main()
