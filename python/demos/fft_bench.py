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
from time import time, perf_counter_ns


rng = np.random.default_rng(42)


def measure_pocketfft(a, nrepeat, nthr, inplace=False):
    import pypocketfft
    times = []
    work = a.copy()
    for i in range(nrepeat):
        if inplace:
            work[()] = a
        inp = work if inplace else a
        t0 = time()
        b = pypocketfft.c2c(inp, out=work, forward=True, nthreads=nthr)
        t1 = time()
        times.append(t1-t0)
    return times, work

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
        else:
            work[()] = 1  # touch the array to page in everything
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


def bench_nd(ndim, nmax, nthr, ntry, tp, funcs, nrepeat, ttl="", filename="",
             nice_sizes=True):
    print("{}D, type {}, max extent is {}:".format(ndim, tp, nmax))
    results = [[] for i in range(len(funcs))]
    for n in range(ntry):
        shp = rng.integers(nmax//3, nmax+1, ndim)
        if nice_sizes:
            shp = np.array([ducc0.fft.good_size(sz) for sz in shp])
        print("  {0:4d}/{1}: shape={2} ...".format(n, ntry, shp), end=" ", flush=True)
        a = (rng.random(shp)-0.5 + 1j*(rng.random(shp)-0.5)).astype(tp)
        output = []
        for func, res in zip(funcs, results):
            tmp = func(a, nrepeat, nthr)
            res.append(np.average(tmp[0]))
            output.append(tmp[1])
            #sleep(1)  # to wait for potential busy-waiting threads
        print("{0:5.2e}/{1:5.2e} = {2:5.2f}  L2 error={3}".format(results[0][n], results[1][n], results[0][n]/results[1][n], ducc0.misc.l2error(output[0], output[1])))
    results = np.array(results)
    import matplotlib.pyplot as plt
    plt.title("{}: {}D, {}, max_extent={}".format(
        ttl, ndim, str(tp), nmax))
    plt.xlabel("time ratio")
    plt.ylabel("counts")
    plt.hist(results[0, :]/results[1, :], bins="auto")
    if filename != "":
        plt.savefig(filename)
    plt.show()
    plt.close()


# Fixed cases for comparing FFT implementations across CPU ISA levels.
# A case uses either C-contiguous layout or F-contiguous (transposed strides).
# Shapes and axes are deterministic so separate builds run identical work.
FFT_CASES = [
    # Small transforms and contiguous 1D entry path.
    ("tiny", "tiny-8", (8,), (0,), "C"),
    ("tiny", "tiny-16", (16,), (0,), "C"),
    ("tiny", "tiny-32", (32,), (0,), "C"),
    ("tiny", "tiny-64", (64,), (0,), "C"),
    ("tiny", "tiny-128", (128,), (0,), "C"),
    ("tiny", "tiny-256", (256,), (0,), "C"),
    # Power-of-two / radix-4 and radix-8 butterflies.
    ("radix", "radix4-256", (256,), (0,), "C"),
    ("radix", "radix8-512", (512,), (0,), "C"),
    ("radix", "radix8-1024", (1024,), (0,), "C"),
    ("radix", "radix8-2048", (2048,), (0,), "C"),
    ("radix", "radix8-4096", (4096,), (0,), "C"),
    ("radix", "radix8-8192", (8192,), (0,), "C"),
    ("radix", "radix8-16384", (16384,), (0,), "C"),
    # Odd and mixed radices, including the 89 x 23 generic-radix case.
    ("generic", "odd-89", (89,), (0,), "C"),
    ("generic", "odd-257", (257,), (0,), "C"),
    ("generic", "odd-511", (511,), (0,), "C"),
    ("generic", "odd-1023", (1023,), (0,), "C"),
    ("generic", "odd-2047", (2047,), (0,), "C"),
    ("generic", "odd-4095", (4095,), (0,), "C"),
    ("generic", "generic-89x23", (89*23,), (0,), "C"),
    ("generic", "mixed-8x89", (8*89,), (0,), "C"),
    ("generic", "mixed-23x89", (23*89,), (0,), "C"),
    # Large 1D FFTs.
    ("long", "long-32768", (32768,), (0,), "C"),
    ("long", "long-65536", (65536,), (0,), "C"),
    ("long", "long-131072", (131072,), (0,), "C"),
    # 2D and 3D transforms.
    ("nd", "nd-16x64", (16,64), (0,1), "C"),
    ("nd", "nd-64x16", (64,16), (0,1), "C"),
    ("nd", "nd-32x128", (32,128), (0,1), "C"),
    ("nd", "nd-64x64", (64,64), (0,1), "C"),
    ("nd", "nd-8x89", (8,89), (0,1), "C"),
    ("nd", "nd-23x89", (23,89), (0,1), "C"),
    ("nd", "nd-64x128", (64,128), (0,1), "C"),
    ("nd", "nd-8x16x32", (8,16,32), (0,1,2), "C"),
    ("nd", "nd-16x16x16", (16,16,16), (0,1,2), "C"),
    ("nd", "nd-16x32x64", (16,32,64), (0,1,2), "C"),
    # Single-axis transforms on strided and transposed arrays.
    ("strided", "strided-axis0-128x32", (128,32), (0,), "C"),
    ("strided", "strided-axis0-89x23", (89,23), (0,), "C"),
    ("strided", "strided-axis1-23x89", (23,89), (1,), "F"),
]


def measure_matrix_case(case, dtype, nrepeat, nwarmup, nthr, seed):
    """Return individual elapsed times (microseconds), excluding setup."""
    _, _, shape, axes, order = case
    local_rng = np.random.default_rng(seed)
    values = (local_rng.random(shape)-0.5
              + 1j*(local_rng.random(shape)-0.5)).astype(dtype)
    inp = np.array(values, order=order, copy=True)
    out = np.empty_like(inp, order="K")

    def execute():
        ducc0.fft.c2c(inp, out=out, axes=axes, forward=True, nthreads=nthr)

    for _ in range(nwarmup):
        execute()
    times = []
    for _ in range(nrepeat):
        start = perf_counter_ns()
        execute()
        times.append((perf_counter_ns()-start)/1000.)
    return times


def run_matrix(args):
    import csv
    import re

    groups = set(args.groups.split(",")) if args.groups != "all" else None
    dtypes = ["complex64", "complex128"] if args.dtype == "both" else [args.dtype]
    cases = [c for c in FFT_CASES
             if (groups is None or c[0] in groups)
             and (not args.case_filter or re.search(args.case_filter, c[1]))]
    if not cases:
        raise SystemExit("No FFT cases matched")
    columns = ["group", "case", "shape", "axes", "layout", "dtype",
               "nthreads", "warmups", "repeats", "median_us", "min_us", "max_us"]
    rows = []
    for case in cases:
        group, name, shape, axes, order = case
        for dtype in dtypes:
            times = measure_matrix_case(case, dtype, args.repeats,
                                        args.warmups, args.threads, args.seed)
            row = [group, name, "x".join(map(str, shape)),
                   ",".join(map(str, axes)), order, dtype,
                   args.threads, args.warmups, args.repeats,
                   float(np.median(times)), min(times), max(times)]
            rows.append(row)
            print(f"{name:27} {dtype:10} {row[9]:11.3f} us", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


def run_legacy():
    f1 = lambda a, nrepeat, nthr: measure_duccfft(a, nrepeat, nthr, inplace=True, noncritical=False)
    f2 = lambda a, nrepeat, nthr: measure_fftw(a, nrepeat, nthr, flags=('FFTW_MEASURE',), timelimit=20)
    funcs = (f1, f2)
    for ndim, nmax in ((1,8192),(2,2048),(3,256)):
        for dtype, suffix in (("c16",""),("c8","_single")):
            bench_nd(ndim, nmax, 1, 10, dtype, funcs, 10,
                     "duccfft/FFTW", f"{ndim}d{suffix}.png", True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DUCC FFT benchmarks")
    parser.add_argument("--legacy", action="store_true",
                        help="run the original randomized DUCC/FFTW comparison")
    parser.add_argument("--groups", default="all",
                        help="comma-separated groups: tiny,radix,generic,long,nd,strided")
    parser.add_argument("--case-filter", default="",
                        help="regex selecting case names")
    parser.add_argument("--dtype", choices=("both","complex64","complex128"), default="both")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", default="", help="optional CSV output")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("repeats and threads must be positive; warmups nonnegative")
    if args.legacy:
        run_legacy()
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
