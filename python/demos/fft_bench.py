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


import numpy as np
import ducc0
from time import perf_counter as time
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)


def measure_fftw(a, nrepeat, nthr,  inplace=False, flags=('FFTW_MEASURE',), timelimit=None):
    import pyfftw
    f1 = pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    f2 = f1 if inplace else pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    fftw = pyfftw.FFTW(f1, f2, flags=flags, axes=range(a.ndim), threads=nthr,
                       planning_timelimit=timelimit)
    f1[()] = a
    times = []
    for i in range(nrepeat):
        t0 = time()
        fftw()
        t1 = time()
        times.append(t1-t0)
    return times, f2

def measure_fftw_np_interface(a, nrepeat, nthr):
    import pyfftw
    pyfftw.interfaces.cache.enable()
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = pyfftw.interfaces.numpy_fft.fftn(a)
        t1 = time()
        del b
        times.append(t1-t0)
    return times, b


def measure_duccfft(a, nrepeat, nthr, inplace=False, noncritical=False):
    times = []
    work = ducc0.misc.make_noncritical(a.copy()) if noncritical else a.copy()
    for i in range(nrepeat):
        if inplace:
            work[()] = a
        inp = work if inplace else a
        t0 = time()
        b = ducc0.fft.c2c(inp, out=work, forward=True, nthreads=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, work


def measure_scipy_fft(a, nrepeat, nthr):
    import scipy.fft
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = scipy.fft.fftn(a, workers=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, b


def measure_numpy_fft(a, nrepeat, nthr):
    if nthr != 1:
        raise NotImplementedError("numpy.fft does not support multiple threads")
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = np.fft.fftn(a)
        t1 = time()
        times.append(t1-t0)
    return times, b


def measure_mkl_fft(a, nrepeat, nthr):
    import os
    os.environ['OMP_NUM_THREADS'] = str(nthr)
    import mkl_fft
    times = []
    b = None
    for i in range(nrepeat):
        del b
        t0 = time()
        b = mkl_fft.fftn(a)
        t1 = time()
        times.append(t1-t0)
    return times, b


def measure_mkl_fft_inplace(a, nrepeat, nthr):
    import os
    os.environ['OMP_NUM_THREADS'] = str(nthr)
    import mkl_fft
    times = []
    b = None
    for i in range(nrepeat):
        b=a.copy()
        t0 = time()
        c = mkl_fft.fftn(b, overwrite_x=True)
        t1 = time()
        times.append(t1-t0)
    return times, c


# Keep the original measurement helpers; collect paired timings rather than
# drawing a histogram for each independent group.
def bench_nd(ndim, nmax, nthr, ntry, tp, funcs, nrepeat, rows, profile,
             nice_sizes=True):
    print("{}D, type {}, max extent is {}:".format(ndim, tp, nmax))
    names = ("DUCC", "FFTW", "SciPy", "NumPy")
    for n in range(ntry):
        shp = rng.integers(nmax//3, nmax+1, ndim)
        if nice_sizes:
            shp = np.array([ducc0.fft.good_size(sz) for sz in shp])
        print("  {0:4d}/{1}: shape={2} ...".format(n, ntry, shp),
              end=" ", flush=True)
        a = (rng.random(shp)-0.5 + 1j*(rng.random(shp)-0.5)).astype(tp)
        timings = []
        output = []
        for func in funcs:
            result = func(a, nrepeat, nthr)
            timings.append(float(np.median(result[0])))
            output.append(result[1])

        for idx, name in enumerate(names):
            error = float(ducc0.misc.l2error(output[0], output[idx]))
            tolerance = 2e-5 if tp == "c8" else 1e-11
            if not np.isfinite(error) or error > tolerance:
                raise RuntimeError(
                    f"{profile} {tp} {ndim}D shape={tuple(shp)} "
                    f"{name} L2 error={error:g} > {tolerance:g}")
            rows.append({
                "profile": profile, "dtype": tp, "ndim": ndim,
                "shape": "x".join(map(str, shp)),
                "elements": int(np.prod(shp)), "sample": n,
                "backend": name, "median_s": timings[idx],
                "speedup": timings[idx] / timings[0],
                "result_dtype": str(output[idx].dtype),
                "l2_error": error,
            })
        print("FFTW/DUCC={:.3f} SciPy/DUCC={:.3f} NumPy/DUCC={:.3f}"
              .format(*(value/timings[0] for value in timings[1:])))


def benchmark_profile(profile, output_dir, ntry, nrepeat):
    # Reset the seed in every process so all ISA profiles see identical shapes.
    global rng
    rng = np.random.default_rng(42)
    f1 = lambda a, r, t: measure_duccfft(a, r, t, inplace=False,
                                         noncritical=True)
    f2 = lambda a, r, t: measure_fftw(a, r, t, flags=('FFTW_MEASURE',),
                                     timelimit=2)
    funcs = (f1, f2, measure_scipy_fft, measure_numpy_fft)
    rows = []
    for tp in ("c16", "c8"):
        for ndim, limit in enumerate((8192, 2048, 256), start=1):
            bench_nd(ndim, limit, 1, ntry, tp, funcs, nrepeat, rows, profile)
    with (output_dir / f"measurements_{profile}.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, profiles, output_dir):
    for backend in ("FFTW", "SciPy", "NumPy"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for r, tp in enumerate(("c16", "c8")):
            for c, ndim in enumerate((1, 2, 3)):
                ax = axes[r, c]
                for profile in profiles:
                    points = sorted(
                        (x["elements"], x["speedup"])
                        for x in rows if x["profile"] == profile
                        and x["dtype"] == tp and x["ndim"] == ndim
                        and x["backend"] == backend)
                    if points:
                        ax.plot(*zip(*points), marker="o", linewidth=1.2,
                                label=profile)
                ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xscale("log")
                ax.grid(alpha=0.2)
                ax.set_title(f"{tp}, {ndim}D")
                ax.set_xlabel("FFT input elements")
                if c == 0:
                    ax.set_ylabel("Reference time / DUCC time")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.95), ncol=len(handles))
        fig.suptitle(f"DUCC versus {backend} (above 1 = DUCC faster)")
        note = ("DUCC and FFTW: preallocated out-of-place; "
                "SciPy/NumPy: allocating API. Reference wheels not ISA-matched.")
        fig.text(0.5, 0.02, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.88))
        fig.savefig(output_dir / f"ducc_vs_{backend.lower()}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DUCC FFT comparison")
    parser.add_argument("--output-dir", default="fft-bench-results")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu = ducc0.misc.cpu_info()
    if args.profile is not None:
        if not cpu.get("multiarch") or cpu.get("active_profile") != args.profile:
            raise RuntimeError(f"requested {args.profile}, detected {cpu}")
        benchmark_profile(args.profile, output_dir, args.samples, args.repeats)
        return

    profiles = (cpu["available_profiles"] if cpu.get("multiarch")
                else ["installed"])
    if cpu.get("multiarch"):
        levels = {"x86-64": "1", "x86-64-v3": "3", "x86-64-v4": "4"}
        for profile in profiles:
            env = os.environ.copy()
            env["DUCC0_MAX_PSABI_LEVEL"] = levels[profile]
            subprocess.run([
                sys.executable, __file__, "--output-dir", str(output_dir),
                "--samples", str(args.samples), "--repeats", str(args.repeats),
                "--profile", profile,
            ], env=env, check=True)
    else:
        benchmark_profile("installed", output_dir, args.samples, args.repeats)

    rows = []
    for profile in profiles:
        with (output_dir / f"measurements_{profile}.csv").open(
                newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["elements"] = int(row["elements"])
                row["ndim"] = int(row["ndim"])
                row["speedup"] = float(row["speedup"])
                rows.append(row)
    make_plots(rows, profiles, output_dir)
    versions = {}
    for package in ("ducc0", "numpy", "scipy", "pyFFTW", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    (output_dir / "metadata.json").write_text(json.dumps({
        "cpu_info": cpu, "profiles_tested": profiles,
        "profiles_skipped": [p for p in ("x86-64", "x86-64-v3", "x86-64-v4")
                             if cpu.get("multiarch") and p not in profiles],
        "versions": versions, "seed": 42, "samples": args.samples,
        "repeats": args.repeats, "threads": 1,
        "transform": "forward complex c2c over all axes",
        "allocation": {
            "DUCC": "preallocated out-of-place",
            "FFTW": "preallocated out-of-place",
            "SciPy": "allocating API", "NumPy": "allocating API",
        }, "ratio": "reference median / DUCC median; >1 favors DUCC",
        "reference_isa": "not matched to the forced DUCC profile",
    }, indent=2) + "\n", encoding="utf-8")
    print("Profiles tested:", ", ".join(profiles))
    print("Saved 3 figures, CSV files and metadata in", output_dir)


if __name__ == "__main__":
    main()
