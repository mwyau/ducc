#!/usr/bin/env python3
"""Deterministic, runtime-gated FFT dispatch benchmark for CI."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


LEVEL_NUMBER = {
    "x86-64": 1,
    "x86-64-v2": 2,
    "x86-64-v3": 3,
    "x86-64-v4": 4,
}
PRODUCTION_PROFILES = ("x86-64", "x86-64-v3", "x86-64-v4")
WARMUPS = 5
REPEATS = 21
SEED = 73591


def _complex_input(rng, dtype, shape):
    real_dtype = "float32" if dtype == "complex64" else "float64"
    real = rng.standard_normal(shape).astype(real_dtype)
    imag = rng.standard_normal(shape).astype(real_dtype)
    return (real + 1j * imag).astype(dtype)


def _measure(name, function, warmups=WARMUPS, repeats=REPEATS, calls=1):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter_ns() - start) / calls / 1e6)
    median = statistics.median(samples)
    mad = statistics.median(abs(sample - median) for sample in samples)
    return {"case": name, "median_ms": median, "mad_ms": mad}


def _c2c_case(ducc0, fft, name, array, axes=None, repeats=REPEATS,
              calls=1):
    output = ducc0.misc.make_noncritical(array.copy())

    def execute():
        kwargs = {"forward": True, "inorm": 0, "nthreads": 1}
        if axes is not None:
            kwargs["axes"] = axes
        fft.c2c(array, out=output, **kwargs)

    return _measure(name, execute, repeats=repeats, calls=calls)


def _run_profile(profile):
    import ducc0
    import numpy as np

    info = ducc0.misc.cpu_info()
    if info["fft"]["selected"] != profile:
        raise RuntimeError(
            f"requested {profile}, runtime selected {info['fft']['selected']}")
    if info["selection_cap"] != profile:
        raise RuntimeError(
            f"requested cap {profile}, diagnostic reports "
            f"{info['selection_cap']}")

    fft = ducc0.fft
    rng = np.random.default_rng(SEED)
    results = []
    for dtype, real_dtype, label in (
            ("complex64", "float32", "c64"),
            ("complex128", "float64", "c128")):
        results.append(_c2c_case(
            ducc0, fft, label + "_c2c_1d_524288",
            _complex_input(rng, dtype, (524288,))))
        results.append(_c2c_case(
            ducc0, fft, label + "_c2c_1024x1024",
            _complex_input(rng, dtype, (1024, 1024)), axes=(0, 1)))
        results.append(_c2c_case(
            ducc0, fft, label + "_c2c_32x48x64",
            _complex_input(rng, dtype, (32, 48, 64)), axes=(0, 1, 2)))
        results.append(_c2c_case(
            ducc0, fft, label + "_c2c_prime_1009",
            _complex_input(rng, dtype, (1009,)), repeats=31))
        results.append(_c2c_case(
            ducc0, fft, label + "_c2c_mixed_196608",
            _complex_input(rng, dtype, (196608,))))

        real = rng.standard_normal(524288).astype(real_dtype)
        spectrum = fft.r2c(real, axes=(0,), forward=True, inorm=0,
                           nthreads=1)
        spectrum_output = np.empty_like(spectrum)
        real_output = np.empty_like(real)
        results.append(_measure(
            label + "_r2c_524288",
            lambda: fft.r2c(real, out=spectrum_output, axes=(0,),
                            forward=True, inorm=0, nthreads=1)))
        results.append(_measure(
            label + "_c2r_524288",
            lambda: fft.c2r(spectrum, out=real_output, axes=(0,),
                            lastsize=524288, forward=True, inorm=0,
                            nthreads=1)))

        small = _complex_input(rng, dtype, (64,))
        small_output = np.empty_like(small)
        results.append(_measure(
            label + "_c2c_small_64",
            lambda: fft.c2c(small, out=small_output, forward=True,
                            inorm=0, nthreads=1), calls=1000))

    strided_base = _complex_input(rng, "complex128", (1024, 2048))
    results.append(_c2c_case(
        ducc0, fft, "c128_c2c_strided_1024x1024", strided_base[:, ::2]))
    return {"requested_profile": profile,
            "selected_profile": info["fft"]["selected"],
            "cpu_info": info, "results": results}


def _runnable_profiles(info):
    compiled = set(info["fft"]["compiled"])
    usable = LEVEL_NUMBER[info["usable_level"]]
    runnable = ["x86-64"] if "x86-64" in compiled else []
    skipped = []
    if "x86-64-v2" in compiled:
        skipped.append({"profile": "x86-64-v2",
                        "reason": "not part of the production default set"})
    else:
        skipped.append({"profile": "x86-64-v2",
                        "reason": "not compiled in the production default wheel"})
    for profile in ("x86-64-v3", "x86-64-v4"):
        if profile not in compiled:
            skipped.append({"profile": profile,
                            "reason": "not compiled in this wheel"})
        elif LEVEL_NUMBER[profile] <= usable:
            runnable.append(profile)
        else:
            skipped.append({
                "profile": profile,
                "reason": "compiled, not runnable on this runner; "
                          f"usable_level={info['usable_level']}"})
    return runnable, skipped


def _speedup(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _timing_cell(by_profile, profile):
    value = by_profile.get(profile)
    if value is None:
        return "—"
    return f"{value['median_ms']:.3f} ± {value['mad_ms']:.3f} ms"


def _speedup_cell(value):
    return "—" if value is None else f"{value:.2f}×"


def _markdown(label, info, rows, executed, skipped):
    lines = [
        f"## FFT runtime dispatch benchmark — {label}",
        "",
        f"- Usable CPU level: `{info['usable_level']}`",
        f"- Compiled FFT profiles: `{', '.join(info['fft']['compiled'])}`",
        f"- Profiles benchmarked: `{', '.join(executed)}`",
        "- Times are median ± median absolute deviation; speedups are "
        "within this runner only.",
        "",
        "| Workload | V1 | V3 | V4 | V3 vs V1 | V4 vs V1 | V4 vs V3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        by_profile = row["profiles"]
        speedups = row["speedups"]
        lines.append(
            f"| {row['workload']} | "
            f"{_timing_cell(by_profile, 'x86-64')} | "
            f"{_timing_cell(by_profile, 'x86-64-v3')} | "
            f"{_timing_cell(by_profile, 'x86-64-v4')} | "
            f"{_speedup_cell(speedups['v3_vs_v1'])} | "
            f"{_speedup_cell(speedups['v4_vs_v1'])} | "
            f"{_speedup_cell(speedups['v4_vs_v3'])} |")
    if skipped:
        lines.extend(["", "Skipped profiles:"])
        for item in skipped:
            lines.append(f"- `{item['profile']}`: {item['reason']}")
    return "\n".join(lines) + "\n"


def _run_parent(args):
    import ducc0

    info = ducc0.misc.cpu_info()
    runnable, skipped = _runnable_profiles(info)
    errors = []
    profile_results = {}
    for profile in runnable:
        env = os.environ.copy()
        env["DUCC0_CPU_MAX"] = profile
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", profile]
        result = subprocess.run(command, env=env, capture_output=True,
                                text=True, check=False)
        if result.returncode:
            errors.append({"profile": profile,
                           "returncode": result.returncode,
                           "error": (result.stderr or result.stdout)[-4000:]})
            continue
        try:
            worker = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            errors.append({"profile": profile, "error": str(exc),
                           "output": result.stdout[-4000:]})
            continue
        if worker["selected_profile"] != profile:
            errors.append({"profile": profile,
                           "error": "worker did not select requested profile"})
            continue
        profile_results[profile] = worker

    cases = {}
    for profile, worker in profile_results.items():
        for result in worker["results"]:
            cases.setdefault(result["case"], {})[profile] = {
                "requested_profile": profile,
                "selected_profile": worker["selected_profile"],
                "median_ms": result["median_ms"],
                "mad_ms": result["mad_ms"],
            }

    rows = []
    for name, by_profile in cases.items():
        v1_time = by_profile.get("x86-64", {}).get("median_ms")
        v3_time = by_profile.get("x86-64-v3", {}).get("median_ms")
        v4_time = by_profile.get("x86-64-v4", {}).get("median_ms")
        rows.append({
            "workload": name,
            "profiles": by_profile,
            "speedups": {
                "v3_vs_v1": _speedup(v1_time, v3_time),
                "v4_vs_v1": _speedup(v1_time, v4_time),
                "v4_vs_v3": _speedup(v3_time, v4_time),
            },
        })

    markdown = _markdown(args.label, info, rows, list(profile_results), skipped)
    print(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(markdown)

    if errors:
        for error in errors:
            print(json.dumps(error), file=sys.stderr)
    return 1 if errors or "x86-64" not in profile_results else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PRODUCTION_PROFILES)
    parser.add_argument("--label", default="runner")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_run_profile(args.worker)))
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
