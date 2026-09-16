"""Benchmark runtime-dispatched SHT targets in fresh processes.

This is development validation tooling. Each target runs in a separate Python
process so that the process-static dispatcher sees its requested upper cap.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys


_TARGETS = ("sse2", "avx", "avx512")
_BENCHMARK_CODE = r'''
import json
import os
import statistics
import time

import ducc0
import numpy as np


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return {"median": statistics.median(values), "min": min(values)}


info = ducc0.misc.cpu_dispatch_info()
cases = json.loads(os.environ["DUCC0_SHT_BENCH_CASES"])
warmup = int(os.environ["DUCC0_SHT_BENCH_WARMUP"])
repeats = int(os.environ["DUCC0_SHT_BENCH_REPEATS"])
results = {}
for case in cases:
    name = case["name"]
    lmax = case["lmax"]
    mmax = case["mmax"]
    ntheta = lmax + 2
    nphi = 2 * mmax + 2
    rng = np.random.default_rng(12345)
    alm = (rng.standard_normal((1, nalm(lmax, mmax))) +
           1j * rng.standard_normal((1, nalm(lmax, mmax)))).astype(np.complex128)
    synthesis_kwargs = dict(
        lmax=lmax, mmax=mmax, spin=0, ntheta=ntheta, nphi=nphi,
        nthreads=1, geometry="CC")
    analysis_kwargs = dict(
        lmax=lmax, mmax=mmax, spin=0, nthreads=1, geometry="CC")
    map_value = ducc0.sht.synthesis_2d(alm=alm, **synthesis_kwargs)
    results[name] = {
        "synthesis": measure(
            lambda: ducc0.sht.synthesis_2d(alm=alm, **synthesis_kwargs),
            warmup, repeats),
        "analysis": measure(
            lambda: ducc0.sht.analysis_2d(map=map_value, **analysis_kwargs),
            warmup, repeats),
    }
print(json.dumps({"info": info, "cases": results},
                 sort_keys=True, separators=(",", ":")))
'''


def _run_target(target, cases, warmup, repeats):
    environment = os.environ.copy()
    environment["DUCC0_CPU_MAX"] = target
    environment["DUCC0_SHT_BENCH_CASES"] = json.dumps(cases)
    environment["DUCC0_SHT_BENCH_WARMUP"] = str(warmup)
    environment["DUCC0_SHT_BENCH_REPEATS"] = str(repeats)
    result = subprocess.run(
        [sys.executable, "-c", _BENCHMARK_CODE],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"target {target} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--include-high", action="store_true",
                        help="also benchmark T511")
    args = parser.parse_args()

    cases = [
        {"name": "T42", "lmax": 42, "mmax": 42},
        {"name": "T255", "lmax": 255, "mmax": 255},
    ]
    if args.include_high:
        cases.append({"name": "T511", "lmax": 511, "mmax": 511})

    first = _run_target("sse2", cases, args.warmup, args.repeats)
    info = first["info"]
    if info["mode"] != "dispatch":
        raise RuntimeError("the installed DUCC module was not built in dispatch mode")
    compiled = set(info["compiled"])
    usable = set(info["usable"])
    results = {"sse2": first}
    skipped = []
    for target in _TARGETS[1:]:
        if target not in compiled:
            skipped.append(f"{target}: not compiled")
        elif target not in usable:
            skipped.append(f"{target}: not usable on this runner")
        else:
            results[target] = _run_target(
                target, cases, args.warmup, args.repeats)

    print("target  case   operation  median_s  min_s")
    for target, result in results.items():
        assert result["info"]["selected"]["sht.inner"] == target
        for case in cases:
            for operation in ("synthesis", "analysis"):
                timing = result["cases"][case["name"]][operation]
                print(
                    f"{target:<8} {case['name']:<6} {operation:<10} "
                    f"{timing['median']:.6f} {timing['min']:.6f}")
    if skipped:
        print("skipped targets: " + "; ".join(skipped))


if __name__ == "__main__":
    main()
