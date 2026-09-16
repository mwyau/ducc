#!/usr/bin/env python3
"""Benchmark the DUCC FFT kernels used by NumPy's 1-D FFT backend.

This mirrors the direct-case matrix from the NumPy DUCC FFT investigation:
complex forward/backward FFTs plus real-to-complex and complex-to-real
transforms, in single and double precision, with one DUCC worker thread.

The benchmark intentionally preallocates output arrays. NumPy's gufunc backend
hands DUCC an existing output buffer, so this isolates the DUCC transform path
from Python-side allocation costs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

SIZES = (
    64,
    256,
    1024,
    2048,
    3072,
    3584,
    4095,
    4096,
    4097,
    4608,
    8192,
    16384,
    32768,
    65536,
)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 256)
CASES = (
    ("fft", "complex64"),
    ("fft", "complex128"),
    ("ifft", "complex64"),
    ("ifft", "complex128"),
    ("rfft", "float32"),
    ("rfft", "float64"),
    ("irfft", "complex64"),
    ("irfft", "complex128"),
)
SEED = 20260915


def _shape(batch: int, length: int) -> tuple[int, ...]:
    return (length,) if batch == 1 else (batch, length)


def _loops_for(n: int, batch: int, pilot: float) -> int:
    work = n * batch
    target = 0.015 if work <= 4096 else 0.025 if work <= 65536 else 0.035
    maximum = (
        2000
        if work <= 4096
        else 200
        if work <= 65536
        else 20
        if work <= 1048576
        else 5
    )
    return max(1, min(maximum, int(target / max(pilot, 1e-9))))


def _make_case(np, fft, operation: str, dtype_name: str, n: int, batch: int):
    # Keep case generation deterministic and independent across matrix points.
    op_index = next(i for i, item in enumerate(CASES) if item == (operation, dtype_name))
    rng = np.random.default_rng(SEED + n + batch * 100003 + op_index * 10000019)
    axes = (0,) if batch == 1 else (1,)

    if operation in ("fft", "ifft"):
        dtype = np.dtype(dtype_name)
        shp = _shape(batch, n)
        a = (
            rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
        ).astype(dtype)
        out = np.empty_like(a)
        forward = operation == "fft"

        def call():
            fft.c2c(
                a,
                axes=axes,
                forward=forward,
                inorm=0,
                out=out,
                nthreads=1,
            )

    elif operation == "rfft":
        dtype = np.dtype(dtype_name)
        shp = _shape(batch, n)
        a = rng.standard_normal(shp).astype(dtype)
        out_dtype = np.complex64 if dtype == np.dtype("float32") else np.complex128
        out = np.empty(_shape(batch, n // 2 + 1), dtype=out_dtype)

        def call():
            fft.r2c(
                a,
                axes=axes,
                forward=True,
                inorm=0,
                out=out,
                nthreads=1,
            )

    elif operation == "irfft":
        dtype = np.dtype(dtype_name)
        shp = _shape(batch, n // 2 + 1)
        a = (
            rng.standard_normal(shp) + 1j * rng.standard_normal(shp)
        ).astype(dtype)
        out_dtype = np.float32 if dtype == np.dtype("complex64") else np.float64
        out = np.empty(_shape(batch, n), dtype=out_dtype)

        def call():
            fft.c2r(
                a,
                axes=axes,
                lastsize=n,
                forward=False,
                inorm=0,
                out=out,
                nthreads=1,
                allow_overwriting_input=False,
            )

    else:
        raise ValueError(operation)

    return call, out


def _measure_case(np, fft, operation: str, dtype: str, n: int, batch: int):
    call, out = _make_case(np, fft, operation, dtype, n, batch)

    # Warm the plan/code path before choosing an adaptive loop count.
    call()
    call()
    start = time.perf_counter()
    call()
    pilot = time.perf_counter() - start
    loops = _loops_for(n, batch, pilot)
    samples = 7 if n <= 4096 else 5
    timings = []

    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(loops):
            call()
        timings.append((time.perf_counter() - start) * 1000.0 / loops)

    median_ms = statistics.median(timings)
    best_ms = min(timings)
    mad_ms = statistics.median(abs(value - median_ms) for value in timings)
    return {
        "operation": operation,
        "dtype": dtype,
        "n": n,
        "batch": batch,
        "loops": loops,
        "samples": samples,
        "median_ms": median_ms,
        "best_ms": best_ms,
        "mad_ms": mad_ms,
    }


def run_benchmark(label: str, output: Path) -> None:
    import numpy as np
    import ducc0

    print(f"label={label}")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"processor={platform.processor()}")
    print(f"ducc0={ducc0.__file__}")
    print(f"numpy={np.__version__}")

    rows = []
    total = len(CASES) * len(SIZES) * len(BATCHES)
    index = 0
    for operation, dtype in CASES:
        for n in SIZES:
            for batch in BATCHES:
                index += 1
                row = _measure_case(np, ducc0.fft, operation, dtype, n, batch)
                rows.append(row)
                print(
                    f"[{index:03d}/{total}] {label:7s} "
                    f"{operation:5s} {dtype:10s} n={n:5d} batch={batch:3d} "
                    f"median={row['median_ms']:.6f} ms "
                    f"best={row['best_ms']:.6f} ms "
                    f"mad={row['mad_ms']:.6f} ms"
                )

    payload = {
        "label": label,
        "ducc0": str(ducc0.__file__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "sizes": SIZES,
        "batches": BATCHES,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def _key(row: dict) -> tuple[str, str, int, int]:
    return row["operation"], row["dtype"], row["n"], row["batch"]


def _gmean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _format_summary(avx_payload: dict, avx512_payload: dict) -> str:
    avx = {_key(row): row for row in avx_payload["rows"]}
    avx512 = {_key(row): row for row in avx512_payload["rows"]}
    if avx.keys() != avx512.keys():
        raise RuntimeError("AVX and AVX-512 benchmark matrices differ")

    comparisons = []
    for key in avx:
        avx_ms = avx[key]["median_ms"]
        avx512_ms = avx512[key]["median_ms"]
        ratio = avx_ms / avx512_ms
        comparisons.append(
            {
                "key": key,
                "avx_ms": avx_ms,
                "avx512_ms": avx512_ms,
                "ratio": ratio,
            }
        )

    overall_ratios = [item["ratio"] for item in comparisons]
    lines = [
        "## NumPy-required DUCC FFT: AVX vs AVX-512",
        "",
        "`AVX / AVX-512` above 1.0 means the AVX-512 build was faster.",
        "",
        "| operation | dtype | cases | median ratio | geometric mean | AVX-512 wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for operation, dtype in CASES:
        subset = [
            item["ratio"]
            for item in comparisons
            if item["key"][0] == operation and item["key"][1] == dtype
        ]
        wins = sum(value > 1.0 for value in subset)
        lines.append(
            f"| {operation} | `{dtype}` | {len(subset)} | "
            f"{statistics.median(subset):.3f} | {_gmean(subset):.3f} | "
            f"{wins}/{len(subset)} |"
        )

    lines.extend(
        [
            f"| **overall** |  | **{len(overall_ratios)}** | "
            f"**{statistics.median(overall_ratios):.3f}** | "
            f"**{_gmean(overall_ratios):.3f}** | "
            f"**{sum(value > 1.0 for value in overall_ratios)}/{len(overall_ratios)}** |",
            "",
            "### Largest AVX-512 wins",
            "",
            "| operation | dtype | n | batch | AVX ms | AVX-512 ms | ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for item in sorted(comparisons, key=lambda value: value["ratio"], reverse=True)[:12]:
        operation, dtype, n, batch = item["key"]
        lines.append(
            f"| {operation} | `{dtype}` | {n} | {batch} | "
            f"{item['avx_ms']:.6f} | {item['avx512_ms']:.6f} | "
            f"{item['ratio']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Largest AVX-512 regressions",
            "",
            "| operation | dtype | n | batch | AVX ms | AVX-512 ms | ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(comparisons, key=lambda value: value["ratio"])[:12]:
        operation, dtype, n, batch = item["key"]
        lines.append(
            f"| {operation} | `{dtype}` | {n} | {batch} | "
            f"{item['avx_ms']:.6f} | {item['avx512_ms']:.6f} | "
            f"{item['ratio']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"AVX module: `{avx_payload['ducc0']}`",
            "",
            f"AVX-512 module: `{avx512_payload['ducc0']}`",
            "",
        ]
    )
    return "\n".join(lines)


def compare(avx_path: Path, avx512_path: Path) -> None:
    avx_payload = json.loads(avx_path.read_text(encoding="utf-8"))
    avx512_payload = json.loads(avx512_path.read_text(encoding="utf-8"))
    summary = _format_summary(avx_payload, avx512_payload)
    print(summary)

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as stream:
            stream.write(summary)
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", required=True, type=Path)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--avx", required=True, type=Path)
    compare_parser.add_argument("--avx512", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "run":
        run_benchmark(args.label, args.output)
    else:
        compare(args.avx, args.avx512)


if __name__ == "__main__":
    main()
