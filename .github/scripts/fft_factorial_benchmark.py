#!/usr/bin/env python3
"""Matched FFT inputs, shared reference timings, and one-config DUCC workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter


# NumPy runs first in each shared reference block and supplies the saved
# numerical witness used to validate SciPy, FFTW, and all four DUCC builds.
REFERENCES = ("numpy", "scipy", "fftw")
PROFILE_NAMES = {1: "x86-64", 3: "x86-64-v3", 4: "x86-64-v4"}
MAX_EXTENTS = {1: 8192, 2: 2048, 3: 256}


def make_cases() -> list[dict]:
    cases = []
    for operation, precisions in (
        ("c2c", ("c16", "c8")),
        ("r2c", ("f64", "f32")),
        ("c2r", ("f64", "f32")),
    ):
        for precision in precisions:
            for ndim in (1, 2, 3):
                cases.append({
                    "id": f"{operation}-{precision}-{ndim}D",
                    "operation": operation,
                    "precision": precision,
                    "ndim": ndim,
                    "fixed_shape": None,
                })

    for shape in ((4095,), (4096,), (64, 4095), (64, 4096),
                  (4095, 64), (4096, 64)):
        if len(shape) == 1:
            case_id = f"c2c-c16-1D-{shape[0]}"
        else:
            case_id = f"c2c-c16-2D-{shape[0]}x{shape[1]}"
        cases.append({
            "id": case_id,
            "operation": "c2c",
            "precision": "c16",
            "ndim": len(shape),
            "fixed_shape": shape,
        })
    return cases


CASES = make_cases()
CASE_BY_ID = {case["id"]: case for case in CASES}


def _iter23(n: int, start: int, best: int) -> int:
    value = start
    while value < n:
        value *= 2
    while True:
        if value < n:
            value *= 3
        elif value > n:
            best = min(best, value)
            if value & 1:
                return best
            value >>= 1
        else:
            return n


def good_size_complex(n: int) -> int:
    """Python equivalent of the pinned base's good_size(..., real=False)."""
    if n <= 12:
        return n
    best = 2 * n
    f11 = 1
    while f11 < best:
        f117 = f11
        while f117 < best:
            f1175 = f117
            while f1175 < best:
                best = _iter23(n, f1175, best)
                if best == n:
                    return n
                f1175 *= 5
            f117 *= 7
        f11 *= 11
    return best


def _numpy():
    import numpy as np
    return np


def output_shape(operation: str, full_shape: tuple[int, ...]) -> tuple[int, ...]:
    if operation == "r2c":
        return full_shape[:-1] + (full_shape[-1] // 2 + 1,)
    return full_shape


def make_input(case: dict, sample_index: int):
    np = _numpy()
    case_index = next(i for i, candidate in enumerate(CASES)
                      if candidate["id"] == case["id"])
    rng = np.random.default_rng(42 + case_index * 1009 + sample_index)
    if case["fixed_shape"] is not None:
        # Fixed controls bypass good_size() by design.
        shape = tuple(case["fixed_shape"])
    else:
        nmax = MAX_EXTENTS[case["ndim"]]
        raw = rng.integers(nmax // 3, nmax + 1, case["ndim"])
        shape = tuple(good_size_complex(int(n)) for n in raw)

    if case["precision"] in ("c16", "f64"):
        real_dtype = np.float64
        complex_dtype = np.complex128
    else:
        real_dtype = np.float32
        complex_dtype = np.complex64

    if case["operation"] == "c2c":
        data = (rng.random(shape) - 0.5 +
                1j * (rng.random(shape) - 0.5)).astype(complex_dtype)
    elif case["operation"] == "r2c":
        data = (rng.random(shape) - 0.5).astype(real_dtype)
    else:
        real = (rng.random(shape) - 0.5).astype(real_dtype)
        data = np.fft.rfftn(real, axes=tuple(range(case["ndim"])))
        data = data.astype(complex_dtype)
    return shape, data


def array_sha256(array) -> str:
    np = _numpy()
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def relative_l2_error(actual, expected, chunk_elements: int = 1 << 20) -> float:
    """Compute relative L2 error in bounded memory, including for memmaps."""
    np = _numpy()
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    left = actual.reshape(-1)
    right = expected.reshape(-1)
    error_sq = 0.0
    expected_sq = 0.0
    for start in range(0, left.size, chunk_elements):
        stop = min(start + chunk_elements, left.size)
        a = np.asarray(left[start:stop], dtype=np.complex128 if np.iscomplexobj(actual)
                       or np.iscomplexobj(expected) else np.float64)
        b = np.asarray(right[start:stop], dtype=a.dtype)
        delta = a - b
        error_sq += float(np.vdot(delta, delta).real)
        expected_sq += float(np.vdot(b, b).real)
    if expected_sq == 0.0:
        return 0.0 if error_sq == 0.0 else float("inf")
    return (error_sq / expected_sq) ** 0.5


def tolerance_for(case: dict) -> float:
    return 1e-5 if case["precision"] in ("c8", "f32") else 1e-10


def call_ducc(ducc0, operation: str, data, full_shape: tuple[int, ...], out):
    axes = tuple(range(len(full_shape)))
    if operation == "c2c":
        return ducc0.fft.c2c(data, axes=axes, forward=True, inorm=0,
                             out=out, nthreads=1)
    if operation == "r2c":
        return ducc0.fft.r2c(data, axes=axes, forward=True, inorm=0,
                             out=out, nthreads=1)
    return ducc0.fft.c2r(data, axes=axes, lastsize=full_shape[-1],
                         forward=False, inorm=0, out=out, nthreads=1)


def _result_dtype(case: dict):
    np = _numpy()
    if case["operation"] == "c2c":
        return np.complex128 if case["precision"] == "c16" else np.complex64
    if case["operation"] == "r2c":
        return np.complex128 if case["precision"] == "f64" else np.complex64
    return np.float64 if case["precision"] == "f64" else np.float32


def _reference_transform(reference: str, operation: str, data,
                         full_shape: tuple[int, ...], nthreads: int):
    np = _numpy()
    axes = tuple(range(len(full_shape)))
    if reference == "numpy":
        if operation == "c2c":
            return np.fft.fftn(data, axes=axes)
        if operation == "r2c":
            return np.fft.rfftn(data, axes=axes)
        return np.fft.irfftn(data, s=full_shape, axes=axes, norm="forward")

    import scipy.fft
    if operation == "c2c":
        return scipy.fft.fftn(data, axes=axes, workers=nthreads)
    if operation == "r2c":
        return scipy.fft.rfftn(data, axes=axes, workers=nthreads)
    return scipy.fft.irfftn(data, s=full_shape, axes=axes,
                            workers=nthreads, norm="forward")


def _measure_fftw(case: dict, data, full_shape: tuple[int, ...], nrepeat: int):
    import numpy as np
    import pyfftw

    operation = case["operation"]
    axes = tuple(range(len(full_shape)))
    input_array = pyfftw.empty_aligned(data.shape, dtype=data.dtype)
    output_array = pyfftw.empty_aligned(output_shape(operation, full_shape),
                                        dtype=_result_dtype(case))
    direction = "FFTW_BACKWARD" if operation == "c2r" else "FFTW_FORWARD"
    plan = pyfftw.FFTW(
        input_array, output_array, axes=axes, direction=direction,
        flags=("FFTW_MEASURE",), threads=1, planning_timelimit=5,
        normalise_idft=False)
    input_array[...] = data
    plan.execute()  # Warm-up is outside the timed samples.
    times = []
    for _ in range(nrepeat):
        input_array[...] = data
        start = perf_counter()
        plan.execute()
        times.append(perf_counter() - start)
    return times, output_array, str(output_array.dtype)


def measure_reference_sample(case: dict, sample_index: int, profile: str,
                             nrepeat: int, expected_path: Path) -> tuple[list[dict], str]:
    """Measure all three references once for one profile/case/sample."""
    import numpy as np

    shape, data = make_input(case, sample_index)
    input_sha = array_sha256(data)
    expected = None
    rows = []
    for reference in REFERENCES:
        times: list[float] = []
        output = None
        error = None
        try:
            if reference == "fftw":
                times, output, output_dtype = _measure_fftw(case, data, shape, nrepeat)
            else:
                transform = lambda: _reference_transform(
                    reference, case["operation"], data, shape, nthreads=1)
                transform()  # Warm-up is outside the timed samples.
                for _ in range(nrepeat):
                    start = perf_counter()
                    output = transform()
                    times.append(perf_counter() - start)
                output_dtype = str(output.dtype)

            expected_shape = output_shape(case["operation"], shape)
            if tuple(output.shape) != expected_shape:
                error = f"output shape {tuple(output.shape)} != {expected_shape}"
            if reference == "numpy" and error is None:
                expected = np.asarray(output)
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(expected_path, expected, allow_pickle=False)
            if reference != "numpy" and expected is not None and error is None:
                error_value = relative_l2_error(output, expected)
                if not np.isfinite(error_value) or error_value > tolerance_for(case):
                    error = (f"reference differs from NumPy: L2 {error_value:.9g} "
                             f"> {tolerance_for(case):.9g}")
            elif reference != "numpy" and expected is None and error is None:
                error = "NumPy correctness witness is unavailable"
        except Exception as exc:  # Keep the failure in the raw results.
            error = f"{type(exc).__name__}: {exc}"
            output_dtype = None if output is None else str(output.dtype)

        rows.append({
            "record_type": "reference",
            "reference": reference,
            "profile": profile,
            "case": case["id"],
            "operation": case["operation"],
            "precision": case["precision"],
            "ndim": case["ndim"],
            "sample_index": sample_index,
            "shape": list(shape),
            "input_sha256": input_sha,
            "nrepeat": nrepeat,
            "times_s": times,
            "median_ms": statistics.median(times) * 1000 if times else None,
            "output_shape": list(output.shape) if output is not None else None,
            "output_dtype": output_dtype,
            "correctness": "pass" if error is None else "fail",
            "correctness_error": error,
            "normalization": "unnormalized forward and backward; c2r uses norm=forward",
            "fftw_planning": "FFTW_MEASURE plan created outside timed execution" if reference == "fftw" else None,
            "timing_reuse": "measured once per profile/case/sample and shared across A-H",
        })
        if reference == "numpy" and error is not None:
            expected = None

    if expected is not None:
        expected_path.with_suffix(expected_path.suffix + ".sha256").write_text(
            array_sha256(expected) + "\n", encoding="ascii")
    return rows, input_sha


def _cpu_identity() -> dict:
    model = platform.processor() or None
    flags = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            elif line.lower().startswith("flags") and not flags:
                flags = line.split(":", 1)[1].split()
    return {
        "model": model,
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "features": flags,
        "runner_name": os.environ.get("GITHUB_RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS", platform.system()),
    }


def _profile_state(ducc0, expected_profile: str, native: bool = False) -> dict:
    info = ducc0.misc.cpu_info()
    if native:
        if info.get("multiarch") is not False or info.get("architecture") != "x86-64":
            raise RuntimeError(f"expected a single-ISA native x86-64 build: {info}")
        return {**info, "build_kind": "native",
                "compiled_profiles": [expected_profile],
                "available_profiles": [expected_profile],
                "configured_limit": expected_profile,
                "active_profile": expected_profile}
    if info.get("multiarch") is not True:
        raise RuntimeError(f"expected an upstream multiarch build, got {info}")
    if info.get("architecture") != "x86-64":
        raise RuntimeError(f"unexpected multiarch architecture: {info}")
    if info.get("compiled_profiles") != ["x86-64", "x86-64-v3", "x86-64-v4"]:
        raise RuntimeError(f"all three profiles must be compiled: {info}")
    if expected_profile not in info.get("available_profiles", []):
        raise RuntimeError(f"requested profile is not available on this CPU: {info}")
    if info.get("active_profile") != expected_profile:
        raise RuntimeError(f"active profile does not match {expected_profile}: {info}")
    return info


def worker_main(args) -> int:
    import ducc0
    import numpy as np

    state = _profile_state(ducc0, args.profile, native=args.native)
    ducc0.misc.preallocate_memory(1)
    cpu_identity = _cpu_identity()
    print(json.dumps({"record_type": "worker_ready", "variant": args.variant,
                      "profile": args.profile, "cpu_info": state,
                      "cpu_identity": cpu_identity}), flush=True)

    for line in sys.stdin:
        try:
            command = json.loads(line)
            if command.get("stop"):
                break
            case = CASE_BY_ID[command["case"]]
            sample_index = int(command["sample_index"])
            shape, data = make_input(case, sample_index)
            input_sha = array_sha256(data)
            if input_sha != command["input_sha256"]:
                raise RuntimeError(
                    f"matched input hash differs: {input_sha} != {command['input_sha256']}")

            expected_shape = output_shape(case["operation"], shape)
            out = np.empty(expected_shape, dtype=_result_dtype(case))
            call_ducc(ducc0, case["operation"], data, shape, out)  # Warm-up.
            times = []
            result = out
            for _ in range(args.nrepeat):
                start = perf_counter()
                result = call_ducc(ducc0, case["operation"], data, shape, out)
                times.append(perf_counter() - start)

            correctness = "unavailable"
            error_value = None
            correctness_error = None
            if command.get("expected_path"):
                try:
                    expected = np.load(command["expected_path"], mmap_mode="r",
                                       allow_pickle=False)
                    if tuple(result.shape) != expected_shape:
                        correctness = "fail"
                        correctness_error = (
                            f"DUCC output shape {tuple(result.shape)} != {expected_shape}")
                    elif tuple(expected.shape) != expected_shape:
                        correctness = "fail"
                        correctness_error = (
                            f"NumPy output shape {tuple(expected.shape)} != {expected_shape}")
                    else:
                        error_value = relative_l2_error(result, expected)
                        correctness = (
                            "pass" if np.isfinite(error_value)
                            and error_value <= tolerance_for(case) else "fail")
                        if correctness == "fail":
                            correctness_error = (
                                f"relative L2 {error_value:.9g} exceeds "
                                f"{tolerance_for(case):.9g}")
                except Exception as exc:
                    correctness = "fail"
                    correctness_error = f"{type(exc).__name__}: {exc}"

            print(json.dumps({
                "record_type": "ducc",
                "variant": args.variant,
                "profile": args.profile,
                "case": case["id"],
                "operation": case["operation"],
                "precision": case["precision"],
                "ndim": case["ndim"],
                "sample_index": sample_index,
                "shape": list(shape),
                "input_sha256": input_sha,
                "nrepeat": args.nrepeat,
                "times_s": times,
                "median_ms": statistics.median(times) * 1000,
                "output_shape": list(result.shape),
                "output_dtype": str(result.dtype),
                "correctness": correctness,
                "l2_error": error_value,
                "tolerance": tolerance_for(case),
                "correctness_error": correctness_error,
                "compiled_profiles": state["compiled_profiles"],
                "available_profiles": state["available_profiles"],
                "configured_profile_limit": state["configured_limit"],
                "active_profile": state["active_profile"],
                "cpu_info": state,
                "cpu_identity": cpu_identity,
            }, sort_keys=True), flush=True)
        except Exception as exc:  # Preserve an explicit failed sample record.
            command_case = locals().get("command", {}).get("case", "unknown")
            print(json.dumps({
                "record_type": "ducc_error",
                "variant": args.variant,
                "profile": args.profile,
                "case": command_case,
                "sample_index": locals().get("sample_index"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "active_profile": state.get("active_profile"),
                "compiled_profiles": state.get("compiled_profiles"),
                "available_profiles": state.get("available_profiles"),
                "configured_profile_limit": state.get("configured_limit"),
                "cpu_info": state,
                "cpu_identity": cpu_identity,
            }, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--profile", choices=tuple(PROFILE_NAMES.values()))
    parser.add_argument("--nrepeat", type=int, default=5)
    args = parser.parse_args()
    if not args.worker:
        parser.error("this module is launched by fft_factorial_run.py")
    if not args.variant or not args.profile:
        parser.error("--worker requires --variant and --profile")
    if args.nrepeat < 1:
        parser.error("--nrepeat must be positive")
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
