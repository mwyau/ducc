# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019-2023 Max-Planck-Society

import argparse
import importlib
import json
import statistics
from time import perf_counter

import numpy as np


ducc0 = None
MAX_EXTENTS = {1: 8192, 2: 2048, 3: 256}
NICE_SIZES = True
REFERENCES = ("fftw", "scipy", "numpy")
VARIANTS = ("baseline", "current", "current-no-lto", "current-lto")
LTO_PROBE_CASE_IDS = (
    "c2c-c16-1D-4095", "c2c-c16-1D-4096",
    "c2c-c16-2D-64x4095", "c2c-c16-2D-64x4096",
    "c2c-c16-2D-4095x64", "c2c-c16-2D-4096x64",
)
PRECISIONS = {
    "c2c": (("c16", np.complex128), ("c8", np.complex64)),
    "r2c": (("f64", np.float64), ("f32", np.float32)),
    "c2r": (("f64", np.float64), ("f32", np.float32)),
}


def make_cases():
    cases = []
    for operation in ("c2c", "r2c", "c2r"):
        for label, dtype in PRECISIONS[operation]:
            for ndim in (1, 2, 3):
                cases.append({
                    "id": "{}-{}-{}D".format(operation, label, ndim),
                    "operation": operation,
                    "precision": label,
                    "dtype": dtype,
                    "ndim": ndim,
                    "fixed_shape": None,
                })
    for shape in ((64, 4095), (4095, 64)):
        cases.append({
            "id": "c2c-c16-2D-{}x{}".format(*shape),
            "operation": "c2c",
            "precision": "c16",
            "dtype": np.dtype(np.complex128),
            "ndim": 2,
            "fixed_shape": shape,
        })
    for shape in ((4095,), (4096,), (64, 4096), (4096, 64)):
        if len(shape) == 1:
            case_id = "c2c-c16-1D-{}".format(shape[0])
        else:
            case_id = "c2c-c16-2D-{}x{}".format(*shape)
        cases.append({
            "id": case_id,
            "operation": "c2c",
            "precision": "c16",
            "dtype": np.dtype(np.complex128),
            "ndim": len(shape),
            "fixed_shape": shape,
        })
    return cases


CASES = make_cases()
CASE_BY_ID = {case["id"]: case for case in CASES}


def output_shape(operation, shape):
    if operation == "r2c":
        return shape[:-1] + (shape[-1] // 2 + 1,)
    if operation == "c2r":
        return shape
    return shape


def make_input(case, sample_index, max_extent=None):
    seed = 42 + CASES.index(case) * 1009 + sample_index
    rng = np.random.default_rng(seed)
    if case["fixed_shape"] is None:
        nmax = max_extent or MAX_EXTENTS[case["ndim"]]
        shape = tuple(int(n) for n in
                      rng.integers(nmax // 3, nmax + 1, case["ndim"]))
        if NICE_SIZES:
            shape = tuple(int(ducc0.fft.good_size(n)) for n in shape)
    else:
        # Historical regression shapes are intentionally not passed through good_size().
        shape = case["fixed_shape"]

    dtype = case["dtype"]
    if case["operation"] == "c2c":
        data = (rng.random(shape) - 0.5 +
                1j * (rng.random(shape) - 0.5)).astype(dtype)
    elif case["operation"] == "r2c":
        data = (rng.random(shape) - 0.5).astype(dtype)
    else:
        real = (rng.random(shape) - 0.5).astype(dtype)
        # Build a valid half spectrum independent of the DUCC variant being measured.
        data = np.fft.rfftn(real, axes=tuple(range(case["ndim"])))
        data = data.astype(np.complex128 if dtype == np.float64 else np.complex64)
    return shape, data


def call_ducc(operation, data, full_shape, out, nthreads):
    axes = tuple(range(len(full_shape)))
    if operation == "c2c":
        return ducc0.fft.c2c(data, axes=axes, forward=True, inorm=0,
                             out=out, nthreads=nthreads)
    if operation == "r2c":
        return ducc0.fft.r2c(data, axes=axes, forward=True, inorm=0,
                             out=out, nthreads=nthreads)
    return ducc0.fft.c2r(data, axes=axes, lastsize=full_shape[-1],
                         forward=False, inorm=0, out=out, nthreads=nthreads)


def measure_ducc(operation, data, full_shape, nrepeat, nthreads, warmup=False):
    if operation == "c2c":
        out_dtype = data.dtype
    elif operation == "r2c":
        out_dtype = np.complex128 if data.dtype == np.float64 else np.complex64
    else:
        out_dtype = np.float64 if data.dtype == np.complex128 else np.float32
    out = np.empty(output_shape(operation, full_shape), dtype=out_dtype)
    if warmup:
        call_ducc(operation, data, full_shape, out, nthreads)
    times = []
    for _ in range(nrepeat):
        t0 = perf_counter()
        result = call_ducc(operation, data, full_shape, out, nthreads)
        times.append(perf_counter() - t0)
    return times, result


def measure_fftw(operation, data, full_shape, nrepeat, nthreads):
    import pyfftw

    axes = tuple(range(len(full_shape)))
    if operation == "c2c":
        out_dtype = data.dtype
        direction = "FFTW_FORWARD"
    elif operation == "r2c":
        out_dtype = np.complex128 if data.dtype == np.float64 else np.complex64
        direction = "FFTW_FORWARD"
    else:
        out_dtype = np.float64 if data.dtype == np.complex128 else np.float32
        direction = "FFTW_BACKWARD"
    f_in = pyfftw.empty_aligned(data.shape, dtype=data.dtype)
    f_out = pyfftw.empty_aligned(output_shape(operation, full_shape),
                                 dtype=out_dtype)
    plan = pyfftw.FFTW(
        f_in, f_out, axes=axes, direction=direction,
        flags=("FFTW_MEASURE",), threads=nthreads, planning_timelimit=5)
    times = []
    for _ in range(nrepeat):
        f_in[...] = data
        t0 = perf_counter()
        plan.execute()
        times.append(perf_counter() - t0)
    return times, f_out


def measure_scipy(operation, data, full_shape, nrepeat, nthreads):
    import scipy.fft

    axes = tuple(range(len(full_shape)))
    transform = {
        "c2c": lambda: scipy.fft.fftn(data, axes=axes, workers=nthreads),
        "r2c": lambda: scipy.fft.rfftn(data, axes=axes, workers=nthreads),
        "c2r": lambda: scipy.fft.irfftn(
            data, s=full_shape, axes=axes, workers=nthreads, norm="forward"),
    }[operation]
    times = []
    result = None
    for _ in range(nrepeat):
        t0 = perf_counter()
        result = transform()
        times.append(perf_counter() - t0)
    return times, result


def measure_numpy(operation, data, full_shape, nrepeat, nthreads):
    if nthreads != 1:
        raise ValueError("numpy.fft does not support multiple threads")
    axes = tuple(range(len(full_shape)))
    transform = {
        "c2c": lambda: np.fft.fftn(data, axes=axes),
        "r2c": lambda: np.fft.rfftn(data, axes=axes),
        "c2r": lambda: np.fft.irfftn(
            data, s=full_shape, axes=axes, norm="forward"),
    }[operation]
    times = []
    result = None
    for _ in range(nrepeat):
        t0 = perf_counter()
        result = transform()
        times.append(perf_counter() - t0)
    return times, result


MEASURERS = {
    "fftw": measure_fftw,
    "scipy": measure_scipy,
    "numpy": measure_numpy,
}


def run_case(case, args):
    result_by_reference = {name: [] for name in args.references}
    for sample_index in range(args.ntry):
        shape, data = make_input(case, sample_index, args.max_extent)
        ducc_times, ducc_result = measure_ducc(
            case["operation"], data, shape, args.nrepeat, args.threads,
            warmup=case["fixed_shape"] is not None)
        expected_shape = output_shape(case["operation"], shape)
        if ducc_result.shape != expected_shape:
            raise RuntimeError("DUCC output shape {} != {} for {}".format(
                ducc_result.shape, expected_shape, case["id"]))

        for reference in args.references:
            ref_times, ref_result = MEASURERS[reference](
                case["operation"], data, shape, args.nrepeat, args.threads)
            if ref_result.shape != expected_shape:
                raise RuntimeError(
                    "{} output shape {} != {} for {}".format(
                        reference, ref_result.shape, expected_shape, case["id"]))
            error = float(ducc0.misc.l2error(ducc_result, ref_result))
            limit = 1e-5 if case["precision"] in ("f32", "c8") else 1e-10
            if not np.isfinite(error) or error > limit:
                raise RuntimeError(
                    "{} numerical mismatch for {} shape {}: L2 error {} (limit {})".format(
                        reference, case["id"], shape, error, limit))
            ducc_median = float(statistics.median(ducc_times))
            reference_median = float(statistics.median(ref_times))
            result_by_reference[reference].append({
                "shape": list(shape),
                "ducc_median_ms": ducc_median * 1000,
                "reference_median_ms": reference_median * 1000,
                "speedup": reference_median / ducc_median,
                "l2_error": error,
                "ducc_output_dtype": str(ducc_result.dtype),
                "reference_output_dtype": str(ref_result.dtype),
            })
            print("  {} shape={} ducc={:.3f} ms ref={:.3f} ms speedup={:.2f}x L2={:.3g}".format(
                reference, shape, ducc_median * 1000,
                reference_median * 1000, reference_median / ducc_median, error),
                flush=True)

    records = []
    for reference, samples in result_by_reference.items():
        ducc_median_ms = float(statistics.median(
            sample["ducc_median_ms"] for sample in samples))
        reference_median_ms = float(statistics.median(
            sample["reference_median_ms"] for sample in samples))
        ratios = [sample["speedup"] for sample in samples]
        records.append({
            "variant": args.variant,
            "profile": args.profile,
            "reference": reference,
            "case": case["id"],
            "operation": case["operation"],
            "precision": case["precision"],
            "ndim": case["ndim"],
            "shapes": [sample["shape"] for sample in samples],
            "ntry": args.ntry,
            "nrepeat": args.nrepeat,
            "ducc_median_ms": ducc_median_ms,
            "reference_median_ms": reference_median_ms,
            "speedup": float(statistics.median(ratios)),
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
            "l2_error": max(sample["l2_error"] for sample in samples),
            "ducc_output_dtype": samples[0]["ducc_output_dtype"],
            "reference_output_dtypes": sorted({
                sample["reference_output_dtype"] for sample in samples}),
        })
    return records


def run_lto_probe(case, args):
    sample_medians = []
    shapes = []
    errors = []
    for sample_index in range(args.ntry):
        shape, data = make_input(case, sample_index, args.max_extent)
        ducc_times, ducc_result = measure_ducc(
            "c2c", data, shape, args.nrepeat, 1, warmup=True)
        expected = np.fft.fftn(data, axes=tuple(range(len(shape))))
        error = float(ducc0.misc.l2error(ducc_result, expected))
        limit = 1e-10
        if (ducc_result.shape != expected.shape or not np.isfinite(error)
                or error > limit):
            raise RuntimeError(
                "{} numerical mismatch for {} shape {}: L2 error {} (limit {})".format(
                    args.variant, case["id"], shape, error, limit))
        sample_medians.append(float(statistics.median(ducc_times)) * 1000)
        shapes.append(list(shape))
        errors.append(error)

    record = {
        "variant": args.variant,
        "profile": args.profile,
        "reference": "ducc",
        "diagnostic": "lto-regression-probe",
        "case": case["id"],
        "operation": "c2c",
        "precision": "c16",
        "ndim": case["ndim"],
        "shapes": shapes,
        "ntry": args.ntry,
        "nrepeat": args.nrepeat,
        "ducc_median_ms": float(statistics.median(sample_medians)),
        "l2_error": max(errors),
        "ducc_output_dtype": str(ducc_result.dtype),
    }
    print("  DUCC median={:.4f} ms; max L2={:.3g}".format(
        record["ducc_median_ms"], record["l2_error"]), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description="Compare DUCC FFT performance")
    parser.add_argument("--reference", choices=REFERENCES + ("all",),
                        default="fftw")
    parser.add_argument("--case", choices=tuple(CASE_BY_ID))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-lto-probe-cases", action="store_true")
    parser.add_argument("--lto-probe", action="store_true",
                        help="measure one fixed complex128 c2c LTO probe case")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ntry", type=int, default=10)
    parser.add_argument("--nrepeat", type=int, default=10)
    parser.add_argument("--max-extent", type=int,
                        help="reduce random case extents for quick local diagnostics")
    parser.add_argument("--variant",
                        choices=VARIANTS + ("current-lto-forceinline",),
                        default="current")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output-jsonl")
    args = parser.parse_args()

    if args.list_cases:
        for case in CASES:
            print(case["id"])
        return
    if args.list_lto_probe_cases:
        for case_id in LTO_PROBE_CASE_IDS:
            print(case_id)
        return
    if args.max_extent is not None and args.max_extent < 3:
        parser.error("--max-extent must be at least 3")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.ntry < 1:
        parser.error("--ntry must be at least 1")
    if args.nrepeat < 1:
        parser.error("--nrepeat must be at least 1")
    if args.reference in ("numpy", "all") and args.threads != 1:
        parser.error("NumPy reference requires --threads 1")

    global ducc0
    ducc0 = importlib.import_module("ducc0")
    args.references = REFERENCES if args.reference == "all" else (args.reference,)
    profile_info = (ducc0.misc.cpu_info() if hasattr(ducc0.misc, "cpu_info")
                    else {"multiarch": False})
    if args.profile is None:
        args.profile = profile_info.get("active_profile", "non-multiarch")
    print("DUCC CPU info:", json.dumps(profile_info, sort_keys=True))
    ducc0.misc.preallocate_memory(1)

    if args.lto_probe:
        if args.case not in LTO_PROBE_CASE_IDS:
            parser.error("--lto-probe requires one of --list-lto-probe-cases")
        if args.threads != 1:
            parser.error("the focused LTO probe requires --threads 1")
        print("LTO probe; case={}; variant={}; profile={}; ntry={}; nrepeat={}".format(
            args.case, args.variant, args.profile, args.ntry, args.nrepeat),
            flush=True)
        record = run_lto_probe(CASE_BY_ID[args.case], args)
        if args.output_jsonl:
            with open(args.output_jsonl, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        return

    print("References: {}; threads: {}; variant: {}; profile: {}".format(
        ", ".join(args.references), args.threads, args.variant, args.profile))

    cases = [CASE_BY_ID[args.case]] if args.case else CASES
    for case in cases:
        print("{}; ntry={}; nrepeat={}".format(case["id"], args.ntry,
                                                args.nrepeat), flush=True)
        records = run_case(case, args)
        if args.output_jsonl:
            with open(args.output_jsonl, "a", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
        for record in records:
            print("  {} median speedup={:.2f}x, observed {:.2f}x..{:.2f}x, max L2={:.3g}".format(
                record["reference"], record["speedup"], record["ratio_min"],
                record["ratio_max"], record["l2_error"]), flush=True)


if __name__ == "__main__":
    main()
