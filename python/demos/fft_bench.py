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
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
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


# The original benchmark shapes and measurement helpers are retained.
def bench_nd(ndim, nmax, ntry, tp, funcs, nrepeat, profile, rows):
    for sample in range(ntry):
        shape = tuple(ducc0.fft.good_size(int(n)) for n in
                      rng.integers(nmax//3, nmax+1, ndim))
        a = (rng.random(shape)-0.5 + 1j*(rng.random(shape)-0.5)).astype(tp)
        measurements = [func(a, nrepeat, 1) for func in funcs]
        output = measurements[0][1]
        for backend, (_, result) in zip(("ducc", "fftw", "scipy", "numpy"), measurements):
            error = ducc0.misc.l2error(output, result)
            tol = 2e-5 if tp == "c8" else 1e-11
            if not np.isfinite(error) or error > tol:
                raise RuntimeError(f"{profile} {backend} {tp} {shape}: L2 error {error:g}")
        medians = [float(np.median(timings)) for timings, _ in measurements]
        rows.append(dict(profile=profile, dtype=tp, ndim=ndim, sample=sample,
                         shape="x".join(map(str, shape)),
                         elements=int(np.prod(shape)), **dict(zip(
                             ("ducc", "fftw", "scipy", "numpy"), medians))))
        print(f"{profile} {tp} {ndim}D {shape}: "
              f"FFTW={medians[1]/medians[0]:.2f} "
              f"SciPy={medians[2]/medians[0]:.2f} "
              f"NumPy={medians[3]/medians[0]:.2f}", flush=True)


def benchmark(output, profile, ntry, nrepeat):
    funcs = (
        lambda a, n, t: measure_duccfft(a, n, t, inplace=False, noncritical=True),
        lambda a, n, t: measure_fftw(a, n, t, timelimit=2),
        measure_scipy_fft, measure_numpy_fft,
    )
    rows = []
    for tp in ("c16", "c8"):
        for ndim, limit in enumerate((8192, 2048, 256), 1):
            bench_nd(ndim, limit, ntry, tp, funcs, nrepeat, profile, rows)
    with (output / f"{profile}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot(output, profiles):
    rows = []
    for profile in profiles:
        with (output / f"{profile}.csv").open(newline="") as file:
            rows.extend(csv.DictReader(file))
    for backend in ("fftw", "scipy", "numpy"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for i, tp in enumerate(("c16", "c8")):
            for j, ndim in enumerate((1, 2, 3)):
                ax = axes[i, j]
                for profile in profiles:
                    points = sorted((int(r["elements"]), float(r[backend])/float(r["ducc"]))
                                    for r in rows if r["profile"] == profile
                                    and r["dtype"] == tp and int(r["ndim"]) == ndim)
                    if points:
                        ax.plot(*zip(*points), "-o", label=profile)
                ax.axhline(1, color="gray", linestyle="--")
                ax.set_xscale("log")
                ax.set_title(f"{tp}, {ndim}D")
                ax.set_xlabel("Input elements")
                if j == 0:
                    ax.set_ylabel("Reference / DUCC time")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.95), ncol=len(profiles))
        fig.suptitle(f"DUCC vs {backend.upper()} (above 1 favors DUCC)")
        fig.text(0.5, 0.01, "DUCC/FFTW out-of-place, preallocated; SciPy/NumPy allocate. "
                 "Reference ISAs not forced.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.88))
        fig.savefig(output / f"ducc_vs_{backend}.png", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="fft-bench-results")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile", type=int, choices=(1, 3, 4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = ducc0.misc.cpu_info()
    names = {1: "x86-64", 3: "x86-64-v3", 4: "x86-64-v4"}
    if args.profile:
        profile = names[args.profile]
        if not info.get("multiarch") or info.get("active_profile") != profile:
            raise RuntimeError(f"requested {profile}, got {info}")
        benchmark(output, profile, args.samples, args.repeats)
        return

    profiles = info["available_profiles"] if info.get("multiarch") else ["installed"]
    if info.get("multiarch"):
        for level, profile in names.items():
            if profile not in profiles:
                print(f"SKIP {profile}: unsupported CPU", flush=True)
                continue
            env = os.environ.copy()
            env["DUCC0_MAX_PSABI_LEVEL"] = str(level)
            subprocess.run([sys.executable, __file__, "--output-dir", str(output),
                            "--samples", str(args.samples), "--repeats", str(args.repeats),
                            "--profile", str(level)], env=env, check=True)
    else:
        benchmark(output, "installed", args.samples, args.repeats)
    plot(output, profiles)
    (output / "metadata.json").write_text(json.dumps(dict(
        cpu_info=info, profiles=profiles, seed=42,
        samples=args.samples, repeats=args.repeats, threads=1,
        ratio="reference time / DUCC time; >1 favors DUCC",
    ), indent=2) + "\n")
    print("Charts and measurements saved to", output)


if __name__ == "__main__":
    main()
