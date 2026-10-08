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
"""Compare multiarch DUCC FFTs with FFTW, SciPy, and NumPy."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import sysconfig
from pathlib import Path
from time import perf_counter

import ducc0
import numpy as np


PROFILE_NAMES = {1: "x86-64", 3: "x86-64-v3", 4: "x86-64-v4"}
PROFILE_LABELS = {"x86-64": "x86-64-v1", "x86-64-v3": "x86-64-v3",
                  "x86-64-v4": "x86-64-v4"}
FIGURES = ("ducc_vs_fftw.png", "ducc_vs_scipy.png", "ducc_vs_numpy.png")
MAX_EXTENTS = {1: 8192, 2: 2048, 3: 256}
DTYPES = (("complex64", np.complex64), ("complex128", np.complex128))
BACKENDS = ("DUCC", "FFTW", "SciPy", "NumPy")
CSV_FIELDS = (
    "profile", "dtype", "ndim", "sample", "shape", "elements", "backend",
    "repeat", "elapsed_seconds", "median_seconds", "result_dtype",
    "allocation", "l2_error",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="fft_bench_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=4,
                        help="input shapes per dtype and dimensionality")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed calls per backend and input shape")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--size-scale", type=float, default=1.0,
                        help="scale the original per-dimension maximum extents")
    parser.add_argument("--fftw-planning-time-limit", type=float, default=2.0,
                        help="maximum FFTW planning time per input shape")
    parser.add_argument("--worker-profile", type=int, choices=(1, 3, 4),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.threads < 1:
        parser.error("samples, repeats, and threads must be positive")
    if not 0 < args.size_scale <= 1:
        parser.error("size-scale must be greater than 0 and at most 1")
    if args.fftw_planning_time_limit <= 0:
        parser.error("fftw-planning-time-limit must be positive")
    return args


def display_profile(profile):
    if profile in PROFILE_LABELS:
        return PROFILE_LABELS[profile]
    if profile.startswith("installed-"):
        return f"installed build ({profile[len('installed-'):]})"
    return profile


def installed_profile_name(cpu_info):
    architecture = cpu_info.get("architecture") or platform.machine()
    suffix = "".join(
        char if char.isalnum() or char in "-_" else "-"
        for char in architecture.lower()
    )
    return f"installed-{suffix}"


def profile_metadata(args, profile, require_multiarch=True):
    cpu_info = ducc0.misc.cpu_info()
    profile_name = PROFILE_NAMES[profile] if isinstance(profile, int) else profile
    if require_multiarch:
        expected = profile_name
        if cpu_info["architecture"] != "x86-64" or not cpu_info["multiarch"]:
            raise RuntimeError("profile subprocess requires an x86-64 multiarch build")
        if expected not in cpu_info["available_profiles"]:
            raise RuntimeError(f"{expected} is not available on this host: {cpu_info}")
        if cpu_info["active_profile"] != expected:
            raise RuntimeError(
                f"requested {expected}, but cpu_info() selected "
                f"{cpu_info['active_profile']}"
            )

    cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
    compiler = {"command": cxx, "version": "unavailable"}
    try:
        result = subprocess.run(
            shlex.split(cxx) + ["--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        lines = (result.stdout or result.stderr).splitlines()
        if lines:
            compiler["version"] = lines[0]
    except (OSError, subprocess.TimeoutExpired):
        pass

    packages = {}
    for package in ("ducc0", "numpy", "scipy", "pyFFTW", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"

    extents = {
        ndim: max(4, round(limit * args.size_scale))
        for ndim, limit in MAX_EXTENTS.items()
    }
    return {
        "profile": profile_name,
        "cpu_info": cpu_info,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "compiler": compiler,
        "packages": packages,
        "parameters": {
            "seed": args.seed,
            "samples": args.samples,
            "repeats": args.repeats,
            "threads": args.threads,
            "numpy_threads": 1,
            "size_scale": args.size_scale,
            "max_extents": extents,
            "fftw_planning_time_limit_seconds":
                args.fftw_planning_time_limit,
            "axes": "all axes of each input",
            "transform": "forward complex-to-complex",
            "normalization": "unnormalized forward transform",
        },
    }


def timed_calls(call, repeats, prepare=None):
    if prepare is not None:
        prepare()
    warmup_output = call()
    del warmup_output

    elapsed = []
    output = None
    for _ in range(repeats):
        if output is not None:
            del output
        if prepare is not None:
            prepare()
        start = perf_counter()
        output = call()
        elapsed.append(perf_counter() - start)
    return elapsed, output


def measure_ducc(a, repeats, threads):
    work = ducc0.misc.make_noncritical(a.copy())

    def prepare():
        work[...] = a

    def execute():
        return ducc0.fft.c2c(
            work,
            axes=tuple(range(a.ndim)),
            forward=True,
            inorm=0,
            out=work,
            nthreads=threads,
        )

    return timed_calls(execute, repeats, prepare)


def measure_fftw(a, repeats, threads, planning_time_limit):
    import pyfftw

    fft_in = pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    fft_out = pyfftw.empty_aligned(a.shape, dtype=a.dtype)
    plan = pyfftw.FFTW(
        fft_in,
        fft_out,
        axes=tuple(range(a.ndim)),
        flags=("FFTW_MEASURE",),
        threads=threads,
        planning_timelimit=planning_time_limit,
    )

    def prepare():
        fft_in[...] = a

    return timed_calls(plan, repeats, prepare)


def measure_scipy(a, repeats, threads):
    from scipy import fft

    return timed_calls(
        lambda: fft.fftn(a, axes=tuple(range(a.ndim)), norm="backward",
                         workers=threads),
        repeats,
    )


def measure_numpy(a, repeats):
    return timed_calls(
        lambda: np.fft.fftn(a, axes=tuple(range(a.ndim)), norm="backward"),
        repeats,
    )


def benchmark_profile(args, profile, output_dir, require_multiarch=True):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import pyfftw  # noqa: F401
        import scipy.fft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "install matplotlib, scipy, and pyfftw to run the FFT benchmark"
        ) from exc

    metadata = profile_metadata(args, profile, require_multiarch)
    profile_name = metadata["profile"]
    print(f"Benchmarking DUCC profile {display_profile(profile_name)}")
    print(json.dumps(metadata["cpu_info"], indent=2))

    thread_env = str(args.threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = thread_env

    rng = np.random.default_rng(args.seed)
    extents = metadata["parameters"]["max_extents"]
    csv_path = output_dir / f"measurements_{profile_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for dtype_name, dtype in DTYPES:
            for ndim in (1, 2, 3):
                max_extent = extents[ndim]
                for sample in range(args.samples):
                    sizes = rng.integers(max(2, max_extent // 3),
                                         max_extent + 1, size=ndim)
                    shape = tuple(int(ducc0.fft.good_size(int(size)))
                                  for size in sizes)
                    a = (rng.random(shape) - 0.5
                         + 1j * (rng.random(shape) - 0.5)).astype(dtype)
                    elements = int(np.prod(shape, dtype=np.int64))
                    print(
                        f"{profile_name} {dtype_name} {ndim}D "
                        f"sample {sample + 1}/{args.samples}: shape={shape}",
                        flush=True,
                    )

                    ducc_times, ducc_output = measure_ducc(
                        a, args.repeats, args.threads)
                    measurements = {
                        "DUCC": {
                            "times": ducc_times,
                            "output_dtype": str(ducc_output.dtype),
                            "allocation": "preallocated output",
                            "l2_error": 0.0,
                        }
                    }
                    references = (
                        ("FFTW", lambda: measure_fftw(
                            a, args.repeats, args.threads,
                            args.fftw_planning_time_limit),
                         "preallocated output"),
                        ("SciPy", lambda: measure_scipy(
                            a, args.repeats, args.threads),
                         "allocates output"),
                        ("NumPy", lambda: measure_numpy(a, args.repeats),
                         "allocates output"),
                    )
                    for backend, measure, allocation in references:
                        times, result = measure()
                        error = float(ducc0.misc.l2error(ducc_output, result))
                        measurements[backend] = {
                            "times": times,
                            "output_dtype": str(result.dtype),
                            "allocation": allocation,
                            "l2_error": error,
                        }
                        del result

                    ducc_median = statistics.median(measurements["DUCC"]["times"])
                    for backend in BACKENDS:
                        result = measurements[backend]
                        median = statistics.median(result["times"])
                        ratio = ("" if backend == "DUCC" else
                                 f"; reference/DUCC={median / ducc_median:.3f}")
                        print(
                            f"  {backend}: median={median:.6g}s "
                            f"L2={result['l2_error']:.3g}{ratio}",
                            flush=True,
                        )
                        for repeat, elapsed in enumerate(result["times"], 1):
                            writer.writerow({
                                "profile": profile_name,
                                "dtype": dtype_name,
                                "ndim": ndim,
                                "sample": sample,
                                "shape": "x".join(map(str, shape)),
                                "elements": elements,
                                "backend": backend,
                                "repeat": repeat,
                                "elapsed_seconds": f"{elapsed:.9g}",
                                "median_seconds": f"{median:.9g}",
                                "result_dtype": result["output_dtype"],
                                "allocation": result["allocation"],
                                "l2_error": f"{result['l2_error']:.9g}",
                            })
                    del ducc_output

    metadata_path = output_dir / f"metadata_{profile_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")


def collect_results(output_dir, tested_profiles):
    samples = {}
    for profile in tested_profiles:
        path = output_dir / f"measurements_{profile}.csv"
        with path.open(newline="", encoding="utf-8") as csvfile:
            for row in csv.DictReader(csvfile):
                key = (row["profile"], row["dtype"], int(row["ndim"]),
                       int(row["sample"]))
                entry = samples.setdefault(key, {
                    "elements": int(row["elements"]), "times": {}})
                entry["times"][row["backend"]] = float(row["median_seconds"])
    return samples


def make_plots(output_dir, tested_profiles, skipped_profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = collect_results(output_dir, tested_profiles)
    for reference in ("FFTW", "SciPy", "NumPy"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        handles = labels = None
        for row, (dtype_name, _) in enumerate(DTYPES):
            for col, ndim in enumerate((1, 2, 3)):
                ax = axes[row, col]
                for profile in tested_profiles:
                    points = []
                    for (sample_profile, sample_dtype, sample_ndim, sample), entry in samples.items():
                        if (sample_profile != profile or sample_dtype != dtype_name
                                or sample_ndim != ndim):
                            continue
                        times = entry["times"]
                        if "DUCC" in times and reference in times:
                            points.append((entry["elements"], sample,
                                           times[reference] / times["DUCC"]))
                    points.sort(key=lambda point: (point[0], point[1]))
                    if points:
                        ax.plot([point[0] for point in points],
                                [point[2] for point in points],
                                marker="o", linewidth=1.2,
                                label=display_profile(profile))
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.2)
                ax.set_title(f"{dtype_name}, {ndim}D")
                ax.set_xlabel("Input elements")
                if col == 0:
                    ax.set_ylabel("Median time ratio\n(reference / DUCC)")
                current_handles, current_labels = ax.get_legend_handles_labels()
                if handles is None and current_handles:
                    handles, labels = current_handles, current_labels

        note = (
            "Ratios above 1 mean DUCC is faster. DUCC output is preallocated; "
            "SciPy and NumPy allocate output. FFTW planning is outside timing."
        )
        if reference == "FFTW":
            note = (
                "Ratios above 1 mean DUCC is faster. DUCC and FFTW use "
                "preallocated output; FFTW planning is outside timing."
            )
        skip_note = (
            "Skipped profiles: " + ", ".join(skipped_profiles)
            + " (not host-supported; see summary.md)"
            if skipped_profiles else "Skipped profiles: none"
        )
        fig.suptitle(f"DUCC vs {reference}", fontsize=15)
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.94), ncol=len(handles))
        fig.subplots_adjust(top=0.83, bottom=0.2, hspace=0.35, wspace=0.28)
        fig.text(0.5, 0.09, skip_note, ha="center", va="center",
                 fontsize=8, wrap=True)
        fig.text(0.5, 0.045, note, ha="center", va="center",
                 fontsize=8, wrap=True)
        path = output_dir / f"ducc_vs_{reference.lower()}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")


def write_summary(output_dir, args, detection_info, tested_profiles,
                  skipped_profiles):
    metadata = {
        "host_cpu_info": detection_info,
        "profile_status": {
            profile: {
                "status": "tested",
                "measurements": f"measurements_{profile}.csv",
                "metadata": f"metadata_{profile}.json",
            }
            for profile in tested_profiles
        },
        "skipped_profiles": skipped_profiles,
        "figures": list(FIGURES),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")

    lines = [
        "# DUCC FFT benchmark",
        "",
        "## Run settings",
        "",
        f"- Host CPU info: `{json.dumps(detection_info, sort_keys=True)}`",
        f"- Tested profiles: {', '.join(display_profile(p) for p in tested_profiles)}",
        f"- Samples per dtype and dimensionality: {args.samples}",
        f"- Timed calls per sample: {args.repeats}",
        f"- Threads: {args.threads} for DUCC, FFTW, and SciPy; NumPy FFT uses one thread",
        f"- Random seed: {args.seed}",
        f"- FFTW planning limit: {args.fftw_planning_time_limit:g} s per shape",
        "- Transform: forward over all axes, unnormalized",
        "- Timing: one warmup, then the median of timed calls",
        "- Ratio: reference median / DUCC median; values above 1 favor DUCC",
        "",
        "## Skipped profiles",
        "",
    ]
    if skipped_profiles:
        lines.extend(f"- {profile}: {reason}"
                     for profile, reason in skipped_profiles.items())
    else:
        lines.append("- None")
    lines.extend((
        "",
        "## Artifacts",
        "",
        *(f"- `{name}`" for name in FIGURES),
        "- `metadata.json`: host profile detection, status, and output filenames",
        "- `metadata_<profile>.json`: build, toolchain, package, and run details",
        "- `measurements_<profile>.csv`: every timed call and sample median",
        "",
        "Figures label output allocation behavior. FFTW planning is outside "
        "timing; SciPy and NumPy are not ISA-matched to DUCC.",
    ))
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


def run_coordinator(args, output_dir):
    detection_info = ducc0.misc.cpu_info()
    print("Detected DUCC CPU information:", flush=True)
    print(json.dumps(detection_info, indent=2), flush=True)
    expected_profiles = [PROFILE_NAMES[level] for level in (1, 3, 4)]
    if detection_info["architecture"] != "x86-64" or not detection_info["multiarch"]:
        architecture = detection_info.get("architecture") or platform.machine()
        installed_profile = installed_profile_name(detection_info)
        print(
            "The installed DUCC build is not x86-64 multiarch; "
            f"benchmarking the installed build ({architecture}).",
            flush=True,
        )
        benchmark_profile(args, installed_profile, output_dir,
                          require_multiarch=False)
        make_plots(output_dir, [installed_profile], {})
        write_summary(output_dir, args, detection_info, [installed_profile], {})
        return
    if detection_info["compiled_profiles"] != expected_profiles:
        raise RuntimeError(
            f"expected compiled profiles {expected_profiles}, got "
            f"{detection_info['compiled_profiles']}"
        )

    available = set(detection_info["available_profiles"])
    if PROFILE_NAMES[1] not in available:
        raise RuntimeError(f"the host does not report the required v1 profile: {detection_info}")
    tested_profiles = []
    skipped_profiles = {}
    script = str(Path(__file__).resolve())
    for level in (1, 3, 4):
        profile = PROFILE_NAMES[level]
        if profile not in available:
            reason = (
                f"cpu_info() reports available_profiles="
                f"{detection_info['available_profiles']} and detected features="
                f"{detection_info['features']}"
            )
            skipped_profiles[profile] = reason
            print(f"SKIP {profile}: {reason}", flush=True)
            continue

        env = os.environ.copy()
        env["DUCC0_MAX_PSABI_LEVEL"] = str(level)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = str(args.threads)
        command = [
            sys.executable, script,
            "--output-dir", str(output_dir),
            "--seed", str(args.seed),
            "--samples", str(args.samples),
            "--repeats", str(args.repeats),
            "--threads", str(args.threads),
            "--size-scale", str(args.size_scale),
            "--fftw-planning-time-limit",
            str(args.fftw_planning_time_limit),
            "--worker-profile", str(level),
        ]
        print(
            "Starting independent benchmark process for "
            f"{PROFILE_LABELS[profile]}", flush=True
        )
        subprocess.run(command, env=env, check=True)
        tested_profiles.append(profile)
    make_plots(output_dir, tested_profiles, skipped_profiles)
    write_summary(output_dir, args, detection_info, tested_profiles,
                  skipped_profiles)


def clear_previous_results(output_dir):
    profiles = set(PROFILE_NAMES.values())
    profiles.add(installed_profile_name(ducc0.misc.cpu_info()))
    for profile in profiles:
        for name in (f"measurements_{profile}.csv", f"metadata_{profile}.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    for name in (*FIGURES, "metadata.json", "summary.md"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_profile is not None:
        benchmark_profile(args, args.worker_profile, output_dir)
    else:
        clear_previous_results(output_dir)
        run_coordinator(args, output_dir)


if __name__ == "__main__":
    main()
