"""Reproducible, single-thread DUCC 1D, native ND and long-FFT policy sweep.

Call through fft_bench.py --policy-suite after installing the selected DUCC wheel.
All cases are identical between profiles, and all timing repeats are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import ducc0

LENGTHS_1D = (512, 1023, 1024, 2047, 2048, 4095, 4096, 8192)
ND2 = ((16, 1024), (32, 512), (64, 256), (128, 128),
       (256, 256), (1024, 16), (63, 256))
ND3 = ((8, 32, 128), (16, 16, 64), (16, 32, 64),
       (32, 32, 32), (64, 32, 16))
LONG = (8192, 16384, 32768, 65536, 131072, 262144, 524288)
BOUNDARY = (1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64)
LONG_BATCH = (1, 2, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
              63, 64, 65, 127, 128, 129, 255, 256, 511, 512, 1024)
MAX_INPUT_BYTES = 128 * 1024 * 1024


def cases():
    seen = set()
    def add(group, shape, batch, dtype, layout, output, forward):
        c = (group, tuple(shape), batch, dtype, layout, output, bool(forward))
        if c not in seen:
            seen.add(c)
            return c
        return None
    for n in LENGTHS_1D:
        for batch in range(1, 65):
            for dtype in ("complex64", "complex128"):
                for output in ("new", "fixed"):
                    c = add("1d", (n,), batch, dtype, "C", output, True)
                    if c: yield c
                    if batch in BOUNDARY:
                        for layout in ("strided", "F"):
                            c = add("1d", (n,), batch, dtype, layout, output, True)
                            if c: yield c
                        c = add("1d", (n,), batch, dtype, "C", output, False)
                        if c: yield c
    for family, shapes in (("2d", ND2), ("3d", ND3)):
        for shape in shapes:
            for batch in BOUNDARY:
                for dtype in ("complex64", "complex128"):
                    for output in ("new", "fixed"):
                        for forward in (True, False):
                            c = add(family, shape, batch, dtype, "C", output, forward)
                            if c: yield c
    for n in LONG:
        for batch in LONG_BATCH:
            if n == 8192 and batch <= 64:
                continue  # Original 1D distribution already covers these.
            if 16 * n * batch > MAX_INPUT_BYTES:
                continue
            for dtype in ("complex64", "complex128"):
                for output in ("new", "fixed"):
                    c = add("long", (n,), batch, dtype, "C", output, True)
                    if c: yield c
                    if batch in (8, 16, 32, 64, 128):
                        c = add("long", (n,), batch, dtype, "C", output, False)
                        if c: yield c


def metadata():
    info = Path("/proc/cpuinfo").read_text()
    lines = info.splitlines()
    model = next((v.split(":", 1)[1].strip() for v in lines
                  if v.startswith("model name")), platform.processor())
    flags = next((v.split(":", 1)[1].strip().split() for v in lines
                  if v.startswith("flags")), [])
    caches = []
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    for entry in sorted(root.glob("index*")):
        try:
            caches.append({k: (entry / k).read_text().strip()
                           for k in ("level", "type", "size", "shared_cpu_list")})
        except OSError:
            pass
    return {
        "cpu_model": model, "cpu_flags": flags, "cache": caches,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "psabi_level": int(ducc0.misc.psabi_level()),
        "python": sys.version, "numpy": np.__version__,
        "ducc": getattr(ducc0, "__version__", None),
        "host": platform.platform(),
        "cases_by_family": {}
    }


def arrays(case):
    group, shape, batch, dtype, layout, output, forward = case
    if group in ("1d", "long") and layout == "strided":
        # FFT axis=0, each transform has stride batch and adjacent lanes
        # traverse separate FFTs (contrasts with contiguous FFT axis).
        fullshape = (shape[0], batch)
        data = np.empty(fullshape, dtype=dtype, order="C")
        axes = (0,)
    else:
        fullshape = (batch,) + tuple(shape)
        data = np.empty(fullshape, dtype=dtype, order="F" if layout == "F" else "C")
        axes = tuple(range(1, len(fullshape)))
    data.fill(0.25 + 0.125j)
    data.flat[0] = complex(0.7, -0.2)
    if output == "fixed":
        out = np.empty_like(data, order="K")
    else:
        out = None
    return data, out, axes


def measure_case(case, samples, sample_ms, correctness=False):
    data, out, axes = arrays(case)
    forward = case[-1]
    kwargs = {"axes": axes, "out": out, "forward": forward, "nthreads": 1}
    fn = lambda: ducc0.fft.c2c(data, **kwargs)
    if correctness:
        got = fn()
        expect = np.fft.fftn(data, axes=axes) if forward else np.fft.ifftn(data, axes=axes)
        if not forward:
            expect = expect * math.prod(data.shape[ax] for ax in axes)
        tol = 5e-4 if case[3] == "complex64" else 2e-10
        np.testing.assert_allclose(got, expect, rtol=tol, atol=tol *
                                   max(1, len(axes)))
    fn()
    fn()
    t0 = time.perf_counter_ns()
    fn()
    initial = max(1, time.perf_counter_ns() - t0)
    iterations = max(1, min(64, int(sample_ms * 1e6 / initial)))
    measurements = []
    for rep in range(samples):
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter_ns() - t0
        measurements.append(elapsed / iterations)
    return measurements, iterations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-ms", type=float, default=2.0)
    parser.add_argument("--suite", choices=("all", "1d", "nd", "long"), default="all")
    parser.add_argument("--correctness-fraction", type=int, default=97,
                        help="Check one case per this many workload definitions")
    args = parser.parse_args(argv)
    if args.samples < 3 or args.sample_ms <= 0:
        parser.error("samples must be >=3 and sample-ms positive")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    meta.update({"profile": args.profile, "samples": args.samples,
                 "sample_ms": args.sample_ms, "suite": args.suite,
                 "max_input_bytes": MAX_INPUT_BYTES})
    with (outdir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    fields = ["profile", "case_id", "family", "shape", "batch", "dtype",
              "layout", "output", "forward", "repeat", "ns", "loops"]
    summary_fields = [k for k in fields if k not in ("repeat", "ns", "loops")]
    summary_fields += ["median_ns", "min_ns", "max_ns", "mad_ns", "repeats", "loops"]
    with (outdir / "raw_repeats.csv").open("w", newline="") as rawf, \
         (outdir / "summary.csv").open("w", newline="") as sumf:
        writer = csv.DictWriter(rawf, fieldnames=fields)
        agg = csv.DictWriter(sumf, fieldnames=summary_fields)
        writer.writeheader()
        agg.writeheader()
        ncases = 0
        byfamily = {}
        for idx, case in enumerate(cases()):
            family, shape, batch, dtype, layout, output, forward = case
            if args.suite != "all" and not (args.suite == family or
                    (args.suite == "nd" and family in ("2d", "3d"))):
                continue
            key = f"{family}:{'x'.join(map(str, shape))}:b{batch}:{dtype}:{layout}:{output}:{int(forward)}"
            check = idx % args.correctness_fraction == 0 if args.correctness_fraction > 0 else False
            times, loops = measure_case(case, args.samples, args.sample_ms, correctness=check)
            record = dict(profile=args.profile, case_id=key, family=family,
                          shape="x".join(map(str, shape)), batch=batch, dtype=dtype,
                          layout=layout, output=output, forward=int(forward))
            for j, ns in enumerate(times):
                writer.writerow({**record, "repeat": j, "ns": ns, "loops": loops})
            med = statistics.median(times)
            agg.writerow({**record, "median_ns": med, "min_ns": min(times),
                          "max_ns": max(times),
                          "mad_ns": statistics.median([abs(t-med) for t in times]),
                          "repeats": len(times), "loops": loops})
            ncases += 1
            byfamily[family] = byfamily.get(family, 0) + 1
            if ncases % 100 == 0:
                print(f"{args.profile}: {ncases} cases complete", flush=True)
                rawf.flush()
                sumf.flush()
    meta["cases_total"] = ncases
    meta["cases_by_family"] = byfamily
    with (outdir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"profile": args.profile, "cases": ncases, "by_family": byfamily}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
