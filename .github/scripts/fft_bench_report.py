#!/usr/bin/env python3

import argparse
import csv
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path


VARIANTS = ("baseline", "current", "current-no-lto")
REFERENCES = ("fftw", "scipy", "numpy")
PROFILES = ("x86-64", "x86-64-v3", "x86-64-v4")


def case_order():
    cases = []
    for operation, precisions in (
            ("c2c", ("c16", "c8")),
            ("r2c", ("f64", "f32")),
            ("c2r", ("f64", "f32"))):
        for precision in precisions:
            cases.extend("{}-{}-{}D".format(operation, precision, ndim)
                         for ndim in (1, 2, 3))
    cases.extend(("c2c-c16-2D-64x4095", "c2c-c16-2D-4095x64"))
    return cases


def read_records(path):
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_metadata(cpu_info_dir):
    cpu_info = {}
    for variant in VARIANTS:
        path = cpu_info_dir / (variant + ".json")
        if path.exists():
            with path.open(encoding="utf-8") as source:
                cpu_info[variant] = json.load(source)
    package_versions = {}
    for package in ("numpy", "scipy", "pyFFTW", "matplotlib"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = "not installed"
    compiler = "unavailable"
    try:
        compiler = subprocess.check_output(
            ["c++", "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    return {
        "runner_os": os.environ.get("FFT_BENCH_RUNNER_OS", platform.platform()),
        "python": platform.python_version(),
        "compiler": compiler,
        "packages": package_versions,
        "cpu_info": cpu_info,
        "source_commits": {
            "baseline": "e41fa307d45b9df465f3429587658e37f3db3b4e",
            "current": os.environ.get("FFT_BENCH_CURRENT_SHA", "unknown"),
            "current-no-lto": os.environ.get("FFT_BENCH_CURRENT_SHA", "unknown"),
        },
    }


def write_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant", "profile", "reference", "case", "ducc_median_ms",
        "reference_median_ms", "speedup", "ratio_min", "ratio_max",
        "l2_error", "ntry", "nrepeat", "shapes",
        "ducc_output_dtype", "reference_output_dtypes",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["shapes"] = json.dumps(row["shapes"], separators=(",", ":"))
            row["reference_output_dtypes"] = ",".join(
                row["reference_output_dtypes"])
            writer.writerow({key: row.get(key, "") for key in fields})


def grouped(records):
    return {(r["reference"], r["profile"], r["variant"], r["case"]): r
            for r in records}


def make_charts(records, chart_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = grouped(records)
    complete_profiles = []
    cases = case_order()
    for profile in PROFILES:
        expected = [(ref, profile, variant, case) for ref in REFERENCES
                    for variant in VARIANTS for case in cases]
        if expected and all(key in by_key for key in expected):
            complete_profiles.append(profile)
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_count = 0
    for reference in REFERENCES:
        for profile in complete_profiles:
            x = np.arange(len(cases), dtype=float)
            width = 0.25
            fig, ax = plt.subplots(figsize=(max(19, len(cases) * 0.95), 7.2))
            for offset, variant in enumerate(VARIANTS):
                values = [by_key[(reference, profile, variant, case)]["speedup"]
                          for case in cases]
                ax.bar(x + (offset - 1) * width, values, width,
                       label=variant)
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
            ax.set_ylabel("reference / DUCC speedup (>1 = DUCC faster)")
            ax.set_title("{} vs DUCC — {}".format(reference.upper(), profile))
            ax.set_xticks(x)
            ax.set_xticklabels(cases, rotation=48, ha="right")
            ax.grid(axis="y", alpha=0.22)
            ax.legend()
            fig.tight_layout()
            fig.savefig(chart_dir / "{}-{}.png".format(reference, profile),
                        dpi=140)
            plt.close(fig)
            chart_count += 1
    return complete_profiles, chart_count


def speedup_cell(record):
    if record is None:
        return "—"
    value = record["speedup"]
    return ("{:.2g}×".format(value) if 0 < value < 0.01
            else "{:.2f}×".format(value))


def table_for_reference(lines, reference, profile, cases, by_key):
    lines.append("## {} — {}".format(reference.upper(), profile))
    lines.append("")
    lines.append("| Case | Baseline | Current | Current no-LTO |")
    lines.append("| --- | ---: | ---: | ---: |")
    for case in cases:
        cells = [speedup_cell(by_key.get((reference, profile, variant, case)))
                 for variant in VARIANTS]
        lines.append("| {} | {} | {} | {} |".format(
            case.replace("-", " "), *cells))
    lines.append("")


def isa_table(lines, cases, by_key, available_profiles):
    lines.extend(("## Current DUCC ISA scaling", ""))
    lines.append("Values are DUCC median time at v1 divided by the target profile; "
                 "`>1.0` means the target profile is faster.")
    lines.append("")
    lines.append("| Case | v3 vs v1 | v4 vs v1 | v4 vs v3 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for case in cases:
        values = []
        for target, source in (("x86-64-v3", "x86-64"),
                               ("x86-64-v4", "x86-64"),
                               ("x86-64-v4", "x86-64-v3")):
            if target not in available_profiles or source not in available_profiles:
                values.append("n/a")
                continue
            target_record = by_key.get(("fftw", target, "current", case))
            source_record = by_key.get(("fftw", source, "current", case))
            if not target_record or not source_record:
                values.append("—")
            else:
                ratio = source_record["ducc_median_ms"] / target_record["ducc_median_ms"]
                values.append("{:.2f}×".format(ratio))
        lines.append("| {} | {} | {} | {} |".format(
            case.replace("-", " "), *values))
    lines.append("")


def numpy_caveat(lines, records):
    rows = {}
    for record in records:
        if record["precision"] not in ("c8", "f32"):
            continue
        key = (record["profile"], record["variant"], record["case"])
        rows.setdefault(key, {})[record["reference"]] = record
    if not any("numpy" in item for item in rows.values()):
        lines.append("NumPy single-precision behavior was not evaluated because "
                     "the results contain no NumPy single-precision cases.")
    else:
        differences = {}
        for (_, _, case), references in rows.items():
            numpy_record = references.get("numpy")
            if numpy_record is None:
                continue
            numpy_dtypes = set(numpy_record["reference_output_dtypes"])
            ducc_dtypes = {numpy_record["ducc_output_dtype"]}
            scipy_record = references.get("scipy")
            scipy_dtypes = (set(scipy_record["reference_output_dtypes"])
                            if scipy_record else set())
            expected = "float32" if case.startswith("c2r-") else "complex64"
            if (numpy_dtypes != {expected} or ducc_dtypes != {expected} or
                    (scipy_dtypes and scipy_dtypes != {expected})):
                differences.setdefault(case, set()).add(
                    "NumPy `{}`, DUCC `{}`, SciPy `{}`".format(
                        "`, `".join(sorted(numpy_dtypes)),
                        "`, `".join(sorted(ducc_dtypes)),
                        "`, `".join(sorted(scipy_dtypes)) or "not recorded"))
        if differences:
            details = ["`{}` ({})".format(case, "; ".join(sorted(values)))
                       for case, values in sorted(differences.items())]
            lines.append("NumPy single-precision caveat: output dtype differed "
                         "from the requested single precision in " +
                         ", ".join(details) + ". Results use the installed "
                         "NumPy behavior without dtype coercion.")
        else:
            lines.append("NumPy single-precision caveat: observed NumPy, DUCC, "
                         "and SciPy output dtypes matched the requested "
                         "single-precision dtypes.")
    lines.append("")


def write_summary(summary_path, records, metadata_info, chart_count,
                  complete_profiles):
    by_key = grouped(records)
    cases = case_order()
    cpu_info = metadata_info["cpu_info"]
    available_by_variant = {
        variant: info.get("available_profiles", [])
        for variant, info in cpu_info.items()
    }
    if available_by_variant:
        available_profiles = [profile for profile in PROFILES
                              if all(profile in values
                                     for values in available_by_variant.values())]
    else:
        available_profiles = []
    cpu_summary = "; ".join(
        "{}: active `{}`, available `{}`".format(
            variant,
            info.get("active_profile", "unknown"),
            "`, `".join(info.get("available_profiles", [])))
        for variant, info in cpu_info.items()) or "unavailable"
    lines = [
        "# FFT benchmark summary",
        "",
        "Metric: `reference_time / ducc_time`",
        "",
        "- `> 1.0`: DUCC faster",
        "- `= 1.0`: equal",
        "- `< 1.0`: reference faster",
        "",
        "Reference comparisons use standard installed FFTW, SciPy, and NumPy "
        "builds on this runner. The ABI label applies only to the forced DUCC "
        "profile; reference libraries were not rebuilt or ISA-matched.",
        "",
        "## Environment",
        "",
        "- Runner OS: `{}`".format(metadata_info["runner_os"]),
        "- Python: `{}`".format(metadata_info["python"]),
        "- Compiler: `{}`".format(metadata_info["compiler"]),
        "- DUCC source commits: baseline `{}`, current/current-no-LTO `{}`".format(
            metadata_info["source_commits"]["baseline"],
            metadata_info["source_commits"]["current"]),
        "- NumPy: `{}`; SciPy: `{}`; pyFFTW: `{}`; Matplotlib: `{}`".format(
            metadata_info["packages"].get("numpy", "unknown"),
            metadata_info["packages"].get("scipy", "unknown"),
            metadata_info["packages"].get("pyFFTW", "unknown"),
            metadata_info["packages"].get("matplotlib", "unknown")),
        "- Compiled DUCC profiles: `x86-64`, `x86-64-v3`, `x86-64-v4`",
        "- Available profiles (common to all builds): `{}`".format(
            "`, `".join(available_profiles) if available_profiles else "unknown"),
        "- CPU/profile detection per DUCC build: " + cpu_summary,
        "",
        "`current` keeps the default CMake IPO/LTO behavior. `current-no-lto` "
        "uses the same source and build settings with only "
        "`DUCC0_ENABLE_LTO=OFF`.",
        "Existing multiarch profile object libraries retain their explicit "
        "IPO-off setting in all three variants; the switch controls IPO for "
        "the extension module target.",
        "",
    ]
    numpy_caveat(lines, records)
    if not records:
        lines.extend(("> No benchmark timing records were produced.", ""))
    observed_profiles = sorted({record["profile"] for record in records},
                               key=lambda profile: PROFILES.index(profile)
                               if profile in PROFILES else len(PROFILES))
    expected_profiles = available_profiles or observed_profiles
    partial = any(
        (reference, profile, variant, case) not in by_key
        for profile in expected_profiles
        for reference in REFERENCES
        for variant in VARIANTS
        for case in cases)
    if partial:
        lines.extend(("> This run produced partial results. Missing entries are "
                      "shown as em dashes, and charts are emitted only for a "
                      "complete reference/profile comparison.", ""))
    for profile in observed_profiles:
        for reference in REFERENCES:
            table_for_reference(lines, reference, profile, cases, by_key)
    isa_table(lines, cases, by_key, available_profiles)
    lines.extend(("## Raw median timings", "",
                  "The artifact includes `tables/timings.csv` with DUCC and "
                  "reference median milliseconds, observed ratio range, L2 "
                  "error, sample shapes, and dtypes for every recorded row.", "",
                  "PNG charts generated: `{}` in the artifact `charts/` directory.".format(
                      chart_count), ""))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Create FFT benchmark charts and Markdown")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-info-dir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    records = read_records(results_path)
    metadata_info = read_metadata(Path(args.cpu_info_dir))
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(parents=True, exist_ok=True)
    write_csv(records, output_dir / "tables" / "timings.csv")
    with (output_dir / "tables" / "environment.json").open(
            "w", encoding="utf-8") as output:
        json.dump(metadata_info, output, indent=2, sort_keys=True)
        output.write("\n")
    complete_profiles, chart_count = make_charts(
        records, output_dir / "charts") if records else ([], 0)
    write_summary(Path(args.summary), records, metadata_info, chart_count,
                  complete_profiles)


if __name__ == "__main__":
    main()
