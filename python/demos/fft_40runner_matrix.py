#!/usr/bin/env python3
"""CPU-labelled, deterministic FFT regression benchmark for two DUCC builds.

Run once per installed DUCC/profile in a fresh Python process. All profiles
use identical case identifiers and inputs. No timing is compared across hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

import ducc0
import numpy as np


def add(cases, suite, kind, shape, dtype, *, axes=None, layout="C",
        threads=1, forward=True, inplace=False, transform_type=2):
    shape = tuple(shape)
    if axes is None:
        axes = tuple(range(len(shape)))
    else:
        axes = tuple(axes)
    desc = dict(suite=suite, kind=kind, shape=shape, dtype=dtype,
                axes=axes, layout=layout, threads=threads,
                forward=forward, inplace=inplace, transform_type=transform_type)
    desc["case_id"] = hashlib.sha256(
        json.dumps(desc, sort_keys=True).encode()
    ).hexdigest()[:20]
    cases.append(desc)


def build_cases(suite):
    cases = []
    if suite == "1d":
        lengths = (1, 2, 3, 4, 5, 7, 8, 11, 16, 23, 29, 31, 32, 33,
                   63, 64, 65, 127, 128, 129, 255, 256, 257, 288,
                   299, 300, 301, 303, 304, 512, 1024, 2048, 4096)
        for dtype in ("complex64", "complex128"):
            for n in lengths:
                for forward in (True, False):
                    add(cases, suite, "c2c", (n,), dtype, forward=forward,
                        inplace=n in (128, 256, 288, 300, 301, 1024))
    elif suite == "batched":
        transitions = ((1, 65536), (2, 32768), (2, 65528),
                       (2, 65536), (3, 65536), (7, 65536),
                       (8, 65536), (9, 65536), (16, 65536),
                       (2, 131064), (2, 131072), (8, 131072),
                       (16, 131072), (32, 131072), (64, 131072),
                       (16, 262144))
        for dtype in ("complex64", "complex128"):
            for batch, n in transitions:
                for threads in (1, 4):
                    add(cases, suite, "c2c", (batch, n), dtype,
                        axes=(1,), threads=threads,
                        inplace=(batch in (2, 8) and n in (65528, 65536)))
    elif suite == "strided":
        shapes = ((3, 127), (7, 256), (8, 288), (9, 301),
                  (9, 1024), (8, 4096), (4, 32768), (8, 65536))
        for dtype in ("complex64", "complex128"):
            for shape in shapes:
                for layout in ("F", "step2"):
                    for axis in (0, 1):
                        add(cases, suite, "c2c", shape, dtype, axes=(axis,),
                            layout=layout, threads=1)
                        if axis == 1:
                            add(cases, suite, "c2c", shape, dtype,
                                axes=(axis,), layout=layout, threads=4)
    elif suite == "nd":
        shapes = ((3, 5), (8, 8), (17, 39), (33, 65),
                  (64, 64), (127, 129), (7, 8, 9),
                  (16, 32, 64), (20, 30, 40), (64, 64, 32))
        for dtype in ("complex64", "complex128"):
            for shape in shapes:
                options = (tuple(range(len(shape))),
                           tuple(reversed(range(len(shape)))),
                           (len(shape) - 1,))
                for axes in options:
                    for threads in (1, 4):
                        add(cases, suite, "c2c", shape, dtype,
                            axes=axes, threads=threads)
    elif suite == "long":
        lengths = (32768, 65528, 65536, 65544, 81920, 92160,
                   98304, 100000, 100352, 131064, 131072,
                   131080, 262144, 262152, 524288)
        for dtype in ("complex64", "complex128"):
            for n in lengths:
                for threads in (1, 4):
                    add(cases, suite, "c2c", (n,), dtype, threads=threads,
                        inplace=n in (65536, 131072))
            for batch, n in ((2, 65536), (8, 65536), (9, 65536),
                             (16, 65536), (16, 131072), (16, 262144)):
                for threads in (1, 4):
                    add(cases, suite, "c2c", (batch, n), dtype,
                        axes=(1,), threads=threads)
    elif suite == "real":
        shapes = ((1,), (2,), (3,), (7,), (8,), (127,), (128,),
                  (255,), (256,), (257,), (1024,), (65536,),
                  (131072,), (3, 5), (8, 16), (9, 127),
                  (16, 64), (4, 8192))
        for dtype in ("float32", "float64"):
            for shape in shapes:
                choices = (tuple(range(len(shape))),)
                if len(shape) == 2:
                    choices += ((1,), (1, 0))
                for axes in choices:
                    for kind in ("r2c", "c2r"):
                        add(cases, suite, kind, shape, dtype, axes=axes,
                            threads=1)
    elif suite == "r2r":
        lengths = (3, 7, 8, 63, 64, 65, 127, 128,
                   129, 255, 256, 257, 288, 300, 301, 1024)
        kinds = ("dct", "dst", "r2r_fftpack", "r2r_fftw",
                 "separable_hartley", "genuine_hartley",
                 "separable_fht", "genuine_fht")
        for dtype in ("float32", "float64"):
            for n in lengths:
                for kind in kinds:
                    if kind in ("dct", "dst"):
                        for t in (1, 2, 3, 4):
                            add(cases, suite, kind, (n,), dtype,
                                transform_type=t)
                    else:
                        add(cases, suite, kind, (n,), dtype)
            for shape in ((7, 8), (16, 32), (7, 8, 9)):
                for kind in ("dct", "dst", "genuine_hartley", "genuine_fht"):
                    add(cases, suite, kind, shape, dtype, threads=1)
    elif suite == "convolution":
        for dtype in ("float32", "float64", "complex64", "complex128"):
            for n in (17, 63, 64, 127, 128, 255, 256, 257,
                      288, 301, 1024, 8192):
                add(cases, suite, "convolve_axis", (n,), dtype, axes=(0,))
                if n <= 1024:
                    add(cases, suite, "convolve_axis", (7, n), dtype,
                        axes=(1,))
    else:
        raise ValueError(f"unknown suite: {suite}")
    ids = [row["case_id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate benchmark case IDs")
    return cases


def cpu_info():
    text = Path("/proc/cpuinfo").read_text(errors="replace") if Path(
        "/proc/cpuinfo").exists() else ""
    model = "unknown"
    flags = set()
    for line in text.splitlines():
        if line.startswith("model name") and model == "unknown":
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
    return dict(cpu_model=model, cpu_arch=platform.machine(),
                cpu_isa="AVX512" if "avx512f" in flags else
                "AVX2" if "avx2" in flags else "baseline",
                has_avx2="avx2" in flags,
                has_avx512="avx512f" in flags,
                cpu_flags=sorted(flags), platform=platform.platform(),
                runner=os.environ.get("RUNNER_NAME", "local"))


def make_array(shape, dtype, layout, rng):
    tp = np.dtype(dtype)
    if tp.kind == "c":
        data = (rng.standard_normal(shape) +
                1j * rng.standard_normal(shape)).astype(tp)
    else:
        data = rng.standard_normal(shape).astype(tp)
    if layout == "C":
        return np.array(data, order="C", copy=True)
    if layout == "F":
        return np.array(data, order="F", copy=True)
    if layout == "step2":
        large = np.empty(tuple(s * 2 for s in shape), dtype=tp)
        view = large[tuple(slice(None, None, 2) for _ in shape)]
        view[...] = data
        assert view.shape == shape
        return view
    raise ValueError(layout)


def reference(case, source, spectrum=None):
    axes = case["axes"]
    kind = case["kind"]
    if kind == "c2c":
        if case["forward"]:
            return np.fft.fftn(source, axes=axes)
        return np.fft.ifftn(source, axes=axes) * np.prod(
            [source.shape[a] for a in axes])
    if kind == "r2c":
        return np.fft.rfftn(source, axes=axes)
    if kind == "c2r":
        sizes = tuple(case["shape"][a] for a in axes)
        return np.fft.irfftn(spectrum, s=sizes, axes=axes) * np.prod(sizes)
    if kind in ("dct", "dst"):
        import scipy.fft
        return (scipy.fft.dctn if kind == "dct" else scipy.fft.dstn)(
            source, type=case["transform_type"], axes=axes)
    if len(case["shape"]) == 1 and kind in (
            "separable_hartley", "genuine_hartley",
            "separable_fht", "genuine_fht"):
        z = np.fft.fft(source)
        return (z.real + z.imag) if kind.endswith("hartley") else (
            z.real - z.imag)
    return None


def prepare(case):
    rng = np.random.default_rng(
        int(case["case_id"], 16) % (2**32) + 4571)
    shape = case["shape"]
    kind = case["kind"]
    source = make_array(shape, case["dtype"], case["layout"], rng)
    if kind == "c2r":
        spectrum = ducc0.fft.r2c(source, axes=case["axes"])
        input_array = np.array(spectrum, copy=True)
        out = np.empty(shape, dtype=case["dtype"])
    else:
        spectrum = None
        input_array = make_array(shape, case["dtype"], case["layout"], rng)
        input_array[...] = source
        out = input_array if case["inplace"] else make_array(
            input_array.shape, case["dtype"], case["layout"], rng)
    fft = ducc0.fft
    threads = case["threads"]
    axes = case["axes"]
    if kind == "c2c":
        def run():
            return fft.c2c(input_array, out=out, axes=axes,
                           forward=case["forward"], nthreads=threads)
    elif kind == "r2c":
        out = np.empty(shape[:-1] + (shape[-1] // 2 + 1,),
                       dtype="complex64" if case["dtype"] == "float32"
                       else "complex128") if axes[-1] == len(shape)-1 else (
            np.empty(tuple(s // 2 + 1 if i == axes[-1] else s
                           for i, s in enumerate(shape)),
                     dtype="complex64" if case["dtype"] == "float32"
                     else "complex128"))
        def run():
            return fft.r2c(input_array, out=out, axes=axes, nthreads=threads)
    elif kind == "c2r":
        def run():
            return fft.c2r(input_array, out=out, axes=axes,
                           lastsize=shape[axes[-1]], forward=False,
                           nthreads=threads)
    elif kind in ("dct", "dst"):
        f = fft.dct if kind == "dct" else fft.dst
        def run():
            return f(input_array, type=case["transform_type"], axes=axes,
                     out=out, nthreads=threads)
    elif kind == "r2r_fftpack":
        def run():
            return fft.r2r_fftpack(input_array, axes, True, False,
                                   out=out, nthreads=threads)
    elif kind == "r2r_fftw":
        def run():
            return fft.r2r_fftw(input_array, axes, True,
                                out=out, nthreads=threads)
    elif kind in ("separable_hartley", "genuine_hartley",
                  "separable_fht", "genuine_fht"):
        f = getattr(fft, kind)
        def run():
            return f(input_array, axes=axes, out=out, nthreads=threads)
    elif kind == "convolve_axis":
        kernel_shape = (min(11, shape[axes[0]]),)
        kernel = make_array(kernel_shape, case["dtype"], "C", rng)
        def run():
            return fft.convolve_axis(input_array, out, axes[0], kernel,
                                     nthreads=threads)
    else:
        raise ValueError(kind)
    return run, input_array, out, source, spectrum


def measure(case, samples, target_ms):
    run, inp, out, source, spectrum = prepare(case)
    got = run()
    assert got is out
    if not np.all(np.isfinite(out)):
        raise AssertionError("non-finite FFT output")
    ref = reference(case, source, spectrum)
    error = None
    if ref is not None:
        error = float(np.linalg.norm(out - ref) /
                      max(1.0e-200, np.linalg.norm(ref)))
        tolerance = 8e-5 if np.dtype(case["dtype"]).itemsize <= 8 else 2e-11
        if error > tolerance:
            raise AssertionError(f"relative error {error:.3g} > {tolerance}")
    for _ in range(2):
        if case["inplace"]:
            inp[...] = source
        run()
    if case["inplace"]:
        iterations = 1
    else:
        started = time.perf_counter_ns()
        run()
        duration = time.perf_counter_ns() - started
        iterations = min(128, max(1, round(target_ms * 1e6 /
                                            max(1000, duration))))
    timings = []
    for _ in range(samples):
        if case["inplace"]:
            inp[...] = source
        started = time.perf_counter_ns()
        for _ in range(iterations):
            run()
        timings.append((time.perf_counter_ns() - started) / iterations)
    return dict(ns_median=statistics.median(timings),
                ns_min=min(timings), ns_max=max(timings),
                ns_samples=timings, iterations=iterations,
                relative_error=error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=(
        "1d", "batched", "strided", "nd", "long", "real", "r2r", "convolution"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cpu = cpu_info()
    expected_level = int(os.environ["DUCC0_MAX_PSABI_LEVEL"])
    actual_level = int(ducc0.misc.psabi_level())
    if actual_level != expected_level:
        raise RuntimeError(
            f"requested v{expected_level}, actual v{actual_level}")
    source_path = str(Path(ducc0.__file__).resolve())
    expected_root = Path(os.environ["DUCC_EXPECTED_INSTALL"]).resolve()
    if not Path(source_path).is_relative_to(expected_root):
        raise RuntimeError(f"module leaked from wrong wheel: {source_path}")
    cases = build_cases(args.suite)
    if args.limit:
        cases = cases[:args.limit]
    metadata = dict(**cpu, suite=args.suite, shard=args.shard,
                    replicate=args.replicate, profile=args.profile,
                    sha=os.environ.get("DUCC_SOURCE_SHA", ""),
                    level=actual_level, module=source_path,
                    case_count=len(cases))
    (out / "metadata.json").write_text(json.dumps(
        metadata, indent=2, sort_keys=True) + "\n")
    failures = []
    with (out / "rows.jsonl").open("w") as f:
        for idx, case in enumerate(cases):
            row = dict(case, profile=args.profile, shard=args.shard,
                       replicate=args.replicate, sha=metadata["sha"],
                       cpu_model=cpu["cpu_model"], cpu_isa=cpu["cpu_isa"],
                       level=actual_level,
                       precision=("single" if np.dtype(
                           case["dtype"]).itemsize <= 8 else "double"))
            # Float64 is 8 bytes, as is complex64. Distinguish via dtype.
            row["precision"] = ("single" if case["dtype"] in
                                ("float32", "complex64") else "double")
            try:
                row.update(measure(case, args.samples, args.target_ms))
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                failures.append(row["case_id"])
                traceback.print_exc()
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            if (idx + 1) % 25 == 0:
                print(f"{args.profile} {args.suite}: "
                      f"{idx + 1}/{len(cases)} cases", flush=True)
    (out / "status.json").write_text(json.dumps(
        dict(**metadata, failures=failures, completed=len(cases)),
        indent=2, sort_keys=True) + "\n")
    print(f"{args.profile} {args.suite}: {len(cases)} cases, "
          f"{len(failures)} failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
