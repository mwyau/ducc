#!/usr/bin/env python3

import argparse
import csv
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path


SAFE_VARIANTS = ("baseline", "current", "current-no-lto")
LTO_VARIANT = "current-lto"
LTO_FORCEINLINE_VARIANT = "current-lto-forceinline"
VARIANTS = SAFE_VARIANTS + (LTO_VARIANT,)
CPU_INFO_VARIANTS = VARIANTS + (LTO_FORCEINLINE_VARIANT,)
REFERENCES = ("fftw", "scipy", "numpy")
PROFILES = ("x86-64", "x86-64-v3", "x86-64-v4")
LTO_PROBE_CASES = (
    "c2c-c16-1D-4095", "c2c-c16-1D-4096",
    "c2c-c16-2D-64x4095", "c2c-c16-2D-64x4096",
    "c2c-c16-2D-4095x64", "c2c-c16-2D-4096x64",
)


def case_order():
    cases = []
    for operation, precisions in (
            ("c2c", ("c16", "c8")),
            ("r2c", ("f64", "f32")),
            ("c2r", ("f64", "f32"))):
        for precision in precisions:
            cases.extend("{}-{}-{}D".format(operation, precision, ndim)
                         for ndim in (1, 2, 3))
    cases.extend(LTO_PROBE_CASES)
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
    for variant in CPU_INFO_VARIANTS:
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
            "current-lto": os.environ.get("FFT_BENCH_CURRENT_SHA", "unknown"),
            LTO_FORCEINLINE_VARIANT: os.environ.get(
                "FFT_BENCH_CURRENT_SHA", "unknown"),
        },
        "workflow_status": {
            "full_lto_build": os.environ.get(
                "FFT_BENCH_FULL_LTO_BUILD_OUTCOME", "unavailable"),
            "full_lto_import": os.environ.get(
                "FFT_BENCH_FULL_LTO_IMPORT_OUTCOME", "unavailable"),
            "full_lto_benchmark": os.environ.get(
                "FFT_BENCH_FULL_LTO_BENCHMARK_OUTCOME", "unavailable"),
            "forceinline_build": os.environ.get(
                "FFT_BENCH_FORCEINLINE_BUILD_OUTCOME", "unavailable"),
            "forceinline_import": os.environ.get(
                "FFT_BENCH_FORCEINLINE_IMPORT_OUTCOME", "unavailable"),
            "forceinline_probe": os.environ.get(
                "FFT_BENCH_FORCEINLINE_PROBE_OUTCOME", "unavailable"),
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


def make_charts(records, chart_dir, lto_built):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = grouped(records)
    cases = case_order()
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_count = 0
    for reference in REFERENCES:
        for profile in PROFILES:
            safe_complete = all(
                (reference, profile, variant, case) in by_key
                for variant in SAFE_VARIANTS for case in cases)
            if not safe_complete:
                continue
            variants = list(SAFE_VARIANTS)
            lto_complete = all(
                (reference, profile, LTO_VARIANT, case) in by_key
                for case in cases)
            if lto_built and lto_complete:
                variants.append(LTO_VARIANT)
            x = np.arange(len(cases), dtype=float)
            width = 0.8 / len(variants)
            fig, ax = plt.subplots(figsize=(max(19, len(cases) * 0.95), 7.2))
            for offset, variant in enumerate(variants):
                values = [by_key[(reference, profile, variant, case)]["speedup"]
                          for case in cases]
                ax.bar(x + (offset - (len(variants) - 1) / 2) * width, values, width,
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
    return chart_count


def speedup_cell(record):
    if record is None:
        return "—"
    value = record["speedup"]
    return ("{:.2g}×".format(value) if 0 < value < 0.01
            else "{:.2f}×".format(value))


def table_for_reference(lines, reference, profile, cases, by_key, variants):
    lines.append("## {} — {}".format(reference.upper(), profile))
    lines.append("")
    lines.append("| Case | {} |".format(" | ".join(variants)))
    lines.append("| --- | {} |".format(" | ".join("---:" for _ in variants)))
    for case in cases:
        cells = [speedup_cell(by_key.get((reference, profile, variant, case)))
                 for variant in variants]
        lines.append("| {} | {} |".format(case.replace("-", " "),
                                         " | ".join(cells)))
    lines.append("")


def any_variant_record(by_key, profile, variant, case):
    for reference in REFERENCES:
        record = by_key.get((reference, profile, variant, case))
        if record is not None:
            return record
    return None


def lto_speedup_cell(no_lto, lto):
    if no_lto is None or lto is None:
        return "—"
    if no_lto.get("shapes") != lto.get("shapes"):
        return "—"
    if lto["ducc_median_ms"] <= 0:
        return "—"
    return "{:.2f}×".format(no_lto["ducc_median_ms"] / lto["ducc_median_ms"])


def lto_table(lines, profiles, cases, by_key):
    lines.extend(("## Full multiarch LTO effect", ""))
    lines.append("LTO speedup = current-no-lto DUCC median time / current-lto DUCC "
                 "median time; `>1.0` means full LTO faster and `<1.0` means "
                 "full LTO slower. Rows compare matched profile/case records "
                 "and the same deterministic sample shapes.")
    lines.append("")
    for profile in profiles:
        lines.append("### {}".format(profile))
        lines.append("")
        lines.append("| Case | LTO speedup |")
        lines.append("| --- | ---: |")
        for case in cases:
            no_lto = any_variant_record(by_key, profile, "current-no-lto", case)
            lto = any_variant_record(by_key, profile, LTO_VARIANT, case)
            lines.append("| {} | {} |".format(
                case.replace("-", " "), lto_speedup_cell(no_lto, lto)))
        lines.append("")


def lto_probe_time(record):
    return "—" if record is None else "{:.4f} ms".format(
        record["ducc_median_ms"])


def lto_probe_ratio(numerator, denominator):
    if (numerator is None or denominator is None or
            numerator.get("shapes") != denominator.get("shapes") or
            denominator["ducc_median_ms"] <= 0):
        return "—"
    return "{:.2f}×".format(
        numerator["ducc_median_ms"] / denominator["ducc_median_ms"])


def lto_regression_table(lines, profiles, by_key, probe_records):
    lines.extend(("## LTO regression probe", ""))
    lines.append(
        "Fixed complex128 c2c shapes bypass `good_size()`. DUCC medians use "
        "matched deterministic inputs and warmed plans. `LTO slowdown` = "
        "current-lto / current-no-lto (`>1` means full LTO is slower). "
        "`Force recovery` = current-lto / current-lto-forceinline "
        "(`>1` means force-inline is faster than full LTO). `Force / no-LTO` "
        "= current-lto-forceinline / current-no-lto (`>1` means the probe "
        "remains slower than no-LTO). The force-inline probe is excluded from "
        "the reference-comparison charts.")
    lines.append("")
    probe_by_key = {
        (record.get("profile"), record.get("case")): record
        for record in probe_records
        if record.get("variant") == LTO_FORCEINLINE_VARIANT
    }
    for profile in profiles:
        lines.append("### {}".format(profile))
        lines.append("")
        lines.append(
            "| Case | current-no-lto DUCC | current-lto DUCC | LTO slowdown | "
            "current-lto-forceinline DUCC | Force recovery | Force / no-LTO |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for case in LTO_PROBE_CASES:
            no_lto = any_variant_record(
                by_key, profile, "current-no-lto", case)
            lto = any_variant_record(by_key, profile, LTO_VARIANT, case)
            forceinline = probe_by_key.get((profile, case))
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                case.replace("-", " "), lto_probe_time(no_lto),
                lto_probe_time(lto), lto_probe_ratio(lto, no_lto),
                lto_probe_time(forceinline),
                lto_probe_ratio(lto, forceinline),
                lto_probe_ratio(forceinline, no_lto)))
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


def write_summary(summary_path, records, probe_records, metadata_info, chart_count):
    by_key = grouped(records)
    cases = case_order()
    cpu_info = metadata_info["cpu_info"]
    profile_order = {profile: index for index, profile in enumerate(PROFILES)}
    observed_profiles = sorted(
        {record["profile"] for record in records},
        key=lambda profile: profile_order.get(profile, len(PROFILES)))
    observed_safe_profiles = sorted(
        {record["profile"] for record in records
         if record["variant"] in SAFE_VARIANTS},
        key=lambda profile: profile_order.get(profile, len(PROFILES)))
    safe_cpu_info = [cpu_info.get(variant) for variant in SAFE_VARIANTS]
    safe_capability_known = all(info is not None for info in safe_cpu_info)
    if safe_capability_known:
        available_profiles = [profile for profile in PROFILES
                              if all(profile in info.get("available_profiles", [])
                                     for info in safe_cpu_info)]
    else:
        available_profiles = observed_safe_profiles
    lto_built = (LTO_VARIANT in cpu_info or
                 any(record["variant"] == LTO_VARIANT for record in records))
    runner_has_v4 = ("x86-64-v4" in available_profiles or
                     "x86-64-v4" in observed_safe_profiles)
    variants = SAFE_VARIANTS + ((LTO_VARIANT,) if lto_built else ())
    forceinline_built = (
        LTO_FORCEINLINE_VARIANT in cpu_info or
        any(record.get("variant") == LTO_FORCEINLINE_VARIANT
            for record in probe_records))
    source_commits = [
        "baseline `{}`".format(metadata_info["source_commits"]["baseline"]),
        "current/current-no-LTO `{}`".format(
            metadata_info["source_commits"]["current"]),
    ]
    if lto_built:
        source_commits.append("current-LTO `{}`".format(
            metadata_info["source_commits"][LTO_VARIANT]))
    if forceinline_built:
        source_commits.append(
            "current-LTO-forceinline `{}` plus the benchmark-only "
            "`detail_fft::special_mul` always-inline annotation".format(
                metadata_info["source_commits"][LTO_FORCEINLINE_VARIANT]))
    source_commit_line = "- DUCC source commits: " + "; ".join(source_commits)
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
        "baseline and current use the existing safe multiarch build model, "
        "where profile object libraries are built without IPO; `current` "
        "keeps the existing default IPO-on setting for the final extension. "
        "`baseline` vs `current` compares fft_tweaks under that model.",
        "`current-lto` is a benchmark-only full-LTO multiarch build with IPO "
        "on for profile objects and the final extension. `current-no-lto` "
        "disables IPO for both. `current-lto` vs `current-no-lto` is the full "
        "multiarch LTO experiment.",
        "",
        "## Environment",
        "",
        "- Runner OS: `{}`".format(metadata_info["runner_os"]),
        "- Python: `{}`".format(metadata_info["python"]),
        "- Compiler: `{}`".format(metadata_info["compiler"]),
        source_commit_line,
        "- Full-LTO build/import/benchmark outcomes: `{}` / `{}` / `{}`".format(
            metadata_info["workflow_status"]["full_lto_build"],
            metadata_info["workflow_status"]["full_lto_import"],
            metadata_info["workflow_status"]["full_lto_benchmark"]),
        "- Force-inline build/import/probe outcomes: `{}` / `{}` / `{}`".format(
            metadata_info["workflow_status"]["forceinline_build"],
            metadata_info["workflow_status"]["forceinline_import"],
            metadata_info["workflow_status"]["forceinline_probe"]),
        "- NumPy: `{}`; SciPy: `{}`; pyFFTW: `{}`; Matplotlib: `{}`".format(
            metadata_info["packages"].get("numpy", "unknown"),
            metadata_info["packages"].get("scipy", "unknown"),
            metadata_info["packages"].get("pyFFTW", "unknown"),
            metadata_info["packages"].get("matplotlib", "unknown")),
        "- Compiled DUCC profiles: `x86-64`, `x86-64-v3`, `x86-64-v4`",
        "- Available profiles (common to safe builds): `{}`".format(
            "`, `".join(available_profiles) if available_profiles else "unknown"),
        "- CPU/profile detection per DUCC build: " + cpu_summary,
        "",
    ]
    if lto_built:
        lines.extend((
            "The full-LTO variant is a diagnostic build and is not ABI-safe "
            "evidence. Its configured v1/v3 dispatch labels indicate the "
            "selected DUCC namespace, not proof that the linked code contains "
            "only that psABI instruction level.",
            "",
        ))
    numpy_caveat(lines, records)
    if not records:
        lines.extend(("> No benchmark timing records were produced.", ""))
    expected_profiles = available_profiles or observed_profiles
    partial = any(
        (reference, profile, variant, case) not in by_key
        for profile in expected_profiles
        for reference in REFERENCES
        for variant in variants
        for case in cases)
    if partial:
        lines.extend(("> This run produced partial results. Missing entries are "
                      "shown as em dashes. Charts are checked independently "
                      "for each reference/profile pair.", ""))
    summary_profiles = available_profiles or observed_profiles
    for profile in summary_profiles:
        for reference in REFERENCES:
            table_for_reference(lines, reference, profile, cases, by_key, variants)
    isa_table(lines, cases, by_key, available_profiles)
    if safe_capability_known and not runner_has_v4:
        lines.extend((
            "## Full multiarch LTO effect",
            "",
            "Full multiarch LTO comparison skipped: this runner does not "
            "support x86-64-v4, and the full-LTO build may contain v4 "
            "instructions outside the v4 dispatch path.",
            "",
        ))
    elif runner_has_v4 and lto_built:
        lto_table(lines, available_profiles or observed_profiles, cases, by_key)
    elif runner_has_v4:
        lines.extend((
            "## Full multiarch LTO effect",
            "",
            "Full multiarch LTO comparison unavailable: current-lto did not "
            "produce CPU-profile metadata or timing records.",
            "",
        ))
    else:
        lines.extend((
            "## Full multiarch LTO effect",
            "",
            "Full multiarch LTO comparison unavailable: runner profile "
            "capability could not be determined from the safe builds.",
            "",
        ))
    if runner_has_v4:
        lto_regression_table(
            lines, available_profiles or observed_safe_profiles,
            by_key, probe_records)
    elif safe_capability_known:
        lines.extend((
            "## LTO regression probe",
            "",
            "Fixed regression/control probe skipped: safe-build CPU detection "
            "did not report x86-64-v4, which is required before importing "
            "either full-LTO multiarch module.",
            "",
        ))
    else:
        lines.extend((
            "## LTO regression probe",
            "",
            "Fixed regression/control probe unavailable because safe-build "
            "CPU capability could not be determined.",
            "",
        ))
    lines.extend(("## Raw median timings", "",
                  "The artifact includes `tables/timings.csv` with DUCC and "
                  "reference aggregate median milliseconds, the primary "
                  "speedup as the median of paired per-shape "
                  "reference/DUCC ratios, observed paired ratio range, L2 "
                  "error, sample shapes, and dtypes. Do not interpret the "
                  "ratio of aggregate timing summaries as the primary "
                  "speedup.", "",
                  "PNG charts generated: `{}` in the artifact `charts/` directory.".format(
                      chart_count), ""))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Create FFT benchmark charts and Markdown")
    parser.add_argument("--results", required=True)
    parser.add_argument("--probe-results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-info-dir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    records = read_records(results_path)
    probe_records = (read_records(Path(args.probe_results))
                     if args.probe_results else [])
    metadata_info = read_metadata(Path(args.cpu_info_dir))
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(parents=True, exist_ok=True)
    write_csv(records, output_dir / "tables" / "timings.csv")
    with (output_dir / "tables" / "environment.json").open(
            "w", encoding="utf-8") as output:
        json.dump(metadata_info, output, indent=2, sort_keys=True)
        output.write("\n")
    lto_built = (LTO_VARIANT in metadata_info["cpu_info"] or
                 any(record["variant"] == LTO_VARIANT for record in records))
    chart_count = (make_charts(records, output_dir / "charts", lto_built)
                   if records else 0)
    write_summary(Path(args.summary), records, probe_records,
                  metadata_info, chart_count)


if __name__ == "__main__":
    main()
