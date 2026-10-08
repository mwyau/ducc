#!/usr/bin/env python3
"""Render complete four-variant Markdown, CSV, and reference/DUCC heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fft_factorial_benchmark import CASES, REFERENCES, PROFILE_NAMES
from prepare_factorial_sources import CONFIGS


DISPLAY_REFERENCES = ("fftw", "scipy", "numpy")
PROFILES = ("x86-64", "x86-64-v3", "x86-64-v4")
VARIANT_IDS = tuple(config["id"] for config in CONFIGS)
MAX_SUMMARY_BYTES = 1_000_000
EXPECTED_NTRY = 3


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def median_or_none(values: list[float]):
    return statistics.median(values) if values else None


def cell(value, digits: int = 3) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    if value >= 1000 or value < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def ratio_cell(value) -> str:
    formatted = cell(value, 2)
    return "—" if formatted == "—" else f"{formatted}×"


def markdown_table(headers: list[str], rows: list[list[str]], align_right=None) -> list[str]:
    if align_right is None:
        align_right = set(range(1, len(headers)))
    align = ["---:" if index in align_right else "---" for index in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(align) + " |"]
    lines.extend("| " + " | ".join(str(item).replace("|", "\\|").replace("\n", " ")
                                  for item in row) + " |" for row in rows)
    return lines


def index_records(timing_records: list[dict], reference_records: list[dict]):
    ducc = {}
    references = {}
    for row in timing_records:
        if row.get("record_type") == "ducc":
            ducc[(row.get("profile"), row.get("case"), row.get("variant"),
                  row.get("sample_index"))] = row
    for row in reference_records:
        if row.get("record_type") == "reference":
            references[(row.get("profile"), row.get("case"), row.get("reference"),
                        row.get("sample_index"))] = row
    return ducc, references


def _paired_samples(left: list[dict], right: list[dict],
                    left_field="median_ms", right_field="median_ms") -> list[float] | None:
    if not left or not right:
        return None
    left_by_sample = {row.get("sample_index"): row for row in left}
    right_by_sample = {row.get("sample_index"): row for row in right}
    if (set(left_by_sample) != set(right_by_sample) or
            len(left_by_sample) != EXPECTED_NTRY):
        return None
    ratios = []
    for sample in sorted(left_by_sample):
        a = left_by_sample[sample]
        b = right_by_sample[sample]
        if any(row.get("correctness", "pass") != "pass" for row in (a, b)):
            return None
        if (a.get("shape") != b.get("shape") or
                a.get("input_sha256") != b.get("input_sha256")):
            return None
        left_value = a.get(left_field)
        right_value = b.get(right_field)
        if (left_value is None or right_value is None or
                not math.isfinite(float(left_value)) or
                not math.isfinite(float(right_value)) or
                float(left_value) <= 0 or float(right_value) <= 0):
            return None
        ratios.append(float(left_value) / float(right_value))
    return ratios


def _rows_for(ducc: dict, profile: str, case_id: str, variant: str) -> list[dict]:
    rows = [row for (p, c, v, _), row in ducc.items()
            if p == profile and c == case_id and v == variant]
    return sorted(rows, key=lambda row: row.get("sample_index", -1))


def ducc_absolute(ducc: dict, profile: str, case_id: str, variant: str):
    rows = _rows_for(ducc, profile, case_id, variant)
    if len(rows) != EXPECTED_NTRY or any(row.get("correctness") != "pass" for row in rows):
        return None
    values = [row.get("median_ms") for row in rows]
    if any(value is None or not math.isfinite(float(value)) or float(value) <= 0
           for value in values):
        return None
    return median_or_none([float(value) for value in values])


def absolute_cell(ducc: dict, profile: str, case_id: str, variant: str) -> str:
    rows = _rows_for(ducc, profile, case_id, variant)
    if any(row.get("correctness") != "pass" for row in rows):
        return "FAIL (correctness)"
    if len(rows) != EXPECTED_NTRY:
        return "INCOMPLETE"
    return cell(ducc_absolute(ducc, profile, case_id, variant), 4)


def factor_ratio(ducc: dict, profile: str, case_id: str,
                 numerator_variant: str, denominator_variant: str):
    ratios = _paired_samples(
        _rows_for(ducc, profile, case_id, numerator_variant),
        _rows_for(ducc, profile, case_id, denominator_variant))
    return median_or_none(ratios) if ratios is not None else None


def reference_ratio(ducc: dict, references: dict, profile: str,
                    case_id: str, variant: str, reference: str):
    drows = _rows_for(ducc, profile, case_id, variant)
    rrows = [row for (p, c, r, _), row in references.items()
             if p == profile and c == case_id and r == reference]
    rrows.sort(key=lambda row: row.get("sample_index", -1))
    ratios = _paired_samples(rrows, drows)
    return median_or_none(ratios) if ratios is not None else None


def scaling_ratio(ducc: dict, numerator_profile: str, denominator_profile: str,
                  case_id: str, variant: str):
    left = [row for (profile, case, v, _), row in ducc.items()
            if profile == numerator_profile and case == case_id and v == variant]
    right = [row for (profile, case, v, _), row in ducc.items()
             if profile == denominator_profile and case == case_id and v == variant]
    left.sort(key=lambda row: row.get("sample_index", -1))
    right.sort(key=lambda row: row.get("sample_index", -1))
    ratios = _paired_samples(left, right)
    return median_or_none(ratios) if ratios is not None else None


def absolute_rows(ducc, profile: str, cases=CASES):
    return [[case["id"], *[
        absolute_cell(ducc, profile, case["id"], variant)
        for variant in VARIANT_IDS]] for case in cases]


def effect_rows(ducc, profile: str, factor: str):
    """Paired speedups for the inline or FFT-tweak factor, conditioned on the other."""
    if factor == "inline":
        pairs = (("A", "B"), ("C", "D"))
    elif factor == "tweaks":
        pairs = (("A", "C"), ("B", "D"))
    else:
        raise ValueError(f"unknown factorial effect: {factor}")
    return [[case["id"], *[
        ratio_cell(factor_ratio(ducc, profile, case["id"], off, on))
        for off, on in pairs]] for case in CASES]


def _paired_ratio_of_factor_ratios(ducc, profile, case_id,
                                   n_off: str, d_off: str,
                                   n_on: str, d_on: str):
    first_num = _rows_for(ducc, profile, case_id, n_off)
    first_den = _rows_for(ducc, profile, case_id, d_off)
    second_num = _rows_for(ducc, profile, case_id, n_on)
    second_den = _rows_for(ducc, profile, case_id, d_on)
    if not all((first_num, first_den, second_num, second_den)):
        return None
    maps = [{row.get("sample_index"): row for row in rows}
            for rows in (first_num, first_den, second_num, second_den)]
    if (len({tuple(sorted(item)) for item in maps}) != 1 or
            len(maps[0]) != EXPECTED_NTRY):
        return None
    ratios = []
    for sample in sorted(maps[0]):
        rows = [item[sample] for item in maps]
        if any(row.get("correctness", "pass") != "pass" for row in rows):
            return None
        if len({tuple(row.get("shape", [])) for row in rows}) != 1 or len(
                {row.get("input_sha256") for row in rows}) != 1:
            return None
        a, b, c, d = [row.get("median_ms") for row in rows]
        if any(value is None or not math.isfinite(float(value)) or float(value) <= 0
               for value in (a, b, c, d)):
            return None
        ratios.append((float(a) / float(b)) / (float(c) / float(d)))
    return median_or_none(ratios)


def interaction_rows(ducc, profile: str):
    """Whether FFT tweaks change the inline-fix benefit; all pairs are matched."""
    rows = []
    for case in CASES:
        cid = case["id"]
        no_tweak = factor_ratio(ducc, profile, cid, "A", "B")
        tweaked = factor_ratio(ducc, profile, cid, "C", "D")
        interaction = _paired_ratio_of_factor_ratios(
            ducc, profile, cid, "C", "D", "A", "B")
        combined = factor_ratio(ducc, profile, cid, "A", "D")
        rows.append([cid, ratio_cell(no_tweak), ratio_cell(tweaked),
                     ratio_cell(interaction), ratio_cell(combined)])
    return rows


def expected_scaling_ratio(ducc, profile_pair, case_id, variant):
    upper, lower = profile_pair
    if upper not in PROFILES or lower not in PROFILES:
        return None
    return scaling_ratio(ducc, lower, upper, case_id, variant)


def dtype_rows(ducc_records, reference_records):
    grouped = defaultdict(lambda: defaultdict(set))
    for row in ducc_records:
        if row.get("record_type") == "ducc" and row.get("output_dtype"):
            grouped[(row.get("operation"), row.get("precision"))]["DUCC"].add(
                row["output_dtype"])
    for row in reference_records:
        if row.get("output_dtype"):
            grouped[(row.get("operation"), row.get("precision"))][
                row.get("reference", "?").upper()].add(row["output_dtype"])
    rows = []
    for key in sorted(grouped):
        values = grouped[key]
        rows.append([f"{key[0]} {key[1]}"] + [
            ", ".join(sorted(values[name])) if values[name] else "—"
            for name in ("DUCC", "FFTW", "SCIPY", "NUMPY")])
    return rows


def build_charts(records_ducc: dict, records_ref: dict, charts_dir: Path,
                 supported_profiles: list[str]) -> list[str]:
    if not records_ducc or not records_ref:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    charts_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for reference in DISPLAY_REFERENCES:
        for profile in PROFILES:
            if profile not in supported_profiles:
                continue
            matrix = np.full((len(CASES), len(VARIANT_IDS)), np.nan)
            for i, case in enumerate(CASES):
                for j, variant in enumerate(VARIANT_IDS):
                    value = reference_ratio(records_ducc, records_ref, profile,
                                            case["id"], variant, reference)
                    if value is not None and value > 0:
                        matrix[i, j] = value
            if not np.isfinite(matrix).any():
                continue
            fig, ax = plt.subplots(figsize=(14, 9))
            shown = np.ma.masked_invalid(matrix)
            image = ax.imshow(shown, aspect="auto", cmap="RdYlGn",
                              norm=TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=4.0))
            ax.set_xticks(range(len(VARIANT_IDS)), VARIANT_IDS)
            ax.set_yticks(range(len(CASES)), [case["id"] for case in CASES], fontsize=7)
            ax.set_xlabel("DUCC source configuration")
            ax.set_ylabel("Operation, precision, dimension, and shape")
            ax.set_title(f"{reference.upper()} time / DUCC time — {profile}\n"
                         ">1 means DUCC is faster; 1.0 is parity")
            for i in range(len(CASES)):
                for j in range(len(VARIANT_IDS)):
                    value = matrix[i, j]
                    label = "—" if not np.isfinite(value) else f"{value:.2f}×"
                    ax.text(j, i, label, ha="center", va="center", fontsize=6)
            colorbar = fig.colorbar(image, ax=ax, pad=0.02)
            colorbar.set_label("Reference / DUCC ratio (white at 1.0 parity)")
            colorbar.set_ticks([0.25, 0.5, 1.0, 2.0, 4.0])
            fig.tight_layout()
            destination = charts_dir / f"{reference}-{profile}.png"
            fig.savefig(destination, dpi=160)
            plt.close(fig)
            outputs.append(str(destination.relative_to(charts_dir.parent)))
    if len(outputs) > 9:
        raise RuntimeError(f"chart limit exceeded: {len(outputs)}")
    return outputs


def write_timing_csv(path: Path, ducc_rows: list[dict], ref_rows: list[dict],
                     supported_profiles: list[str], ntry: int,
                     variants: dict) -> None:
    fields = ["record_type", "measurement_status", "variant", "profile", "reference",
              "case", "operation", "precision", "ndim", "sample_index", "shape",
              "input_sha256", "times_s", "median_ms", "output_shape", "output_dtype",
              "correctness", "l2_error", "tolerance", "correctness_error",
              "compiled_profiles", "available_profiles", "configured_profile_limit",
              "active_profile", "cpu_identity", "cpu_info", "profile_metadata_scope",
              "variant_rotation", "rotation_order_index"]
    rows = list(ref_rows) + list(ducc_rows)
    seen = set()
    for row in rows:
        if row.get("record_type") == "ducc":
            seen.add((row.get("profile"), row.get("case"), row.get("variant"),
                      row.get("sample_index")))
    for profile in supported_profiles:
        for case in CASES:
            for config in CONFIGS:
                variant = config["id"]
                for sample_index in range(ntry):
                    key = (profile, case["id"], variant, sample_index)
                    if key in seen:
                        continue
                    status = variants.get(variant, {})
                    rows.append({
                        "record_type": "ducc", "measurement_status": "unavailable",
                        "variant": variant, "profile": profile,
                        "case": case["id"], "operation": case["operation"],
                        "precision": case["precision"], "ndim": case["ndim"],
                        "sample_index": sample_index,
                        "correctness": "unmeasured",
                        "correctness_error": status.get("build_error") or
                            status.get("validation_error") or
                            status.get("runtime_import_error") or
                            "sample result was not produced",
                    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = dict(row)
            for name in ("shape", "input_sha256", "times_s", "output_shape",
                         "compiled_profiles", "available_profiles", "cpu_identity",
                         "cpu_info"):
                if isinstance(cooked.get(name), (dict, list, tuple)):
                    cooked[name] = json.dumps(cooked[name], separators=(",", ":"))
            writer.writerow({field: cooked.get(field, "") for field in fields})


def add_case_table(lines: list[str], title: str, headers: list[str], rows: list[list[str]],
                   note: str = "") -> None:
    lines.extend([f"### {title}", ""])
    if note:
        lines.extend([note, ""])
    if rows:
        lines.extend(markdown_table(headers, rows))
    else:
        lines.append("No rows were measured.")
    lines.append("")


def make_report(output_dir: Path) -> tuple[str, int]:
    global EXPECTED_NTRY
    status = load_json(output_dir / "run-status.json", {})
    EXPECTED_NTRY = int(status.get("ntry", 3))
    manifest = load_json(output_dir / "configuration-manifest.json", {})
    build_diagnostics = load_json(output_dir / "build-diagnostics.json", {})
    isa_notes = load_json(output_dir / "isa-validation.json", {})
    timings = load_jsonl(output_dir / "timings.jsonl")
    references_raw = load_jsonl(output_dir / "references.jsonl")
    ducc, references = index_records(timings, references_raw)
    native_status = load_json(output_dir / "native-status.json", {})
    native_raw = load_jsonl(output_dir / "native-timings.jsonl")
    native_reference_raw = load_jsonl(output_dir / "native-references.jsonl")
    native_ducc, native_references = index_records(native_raw, native_reference_raw)
    native_profile = native_status.get("selected_native_profile")
    native_ids = tuple(f"N{variant}{lto}" for lto in (0, 1) for variant in VARIANT_IDS)
    combined_ducc = {**ducc, **native_ducc}
    supported = [profile for profile in PROFILES
                 if profile in status.get("supported_profiles", [])]
    skipped = [profile for profile in PROFILES if profile not in supported]

    write_timing_csv(output_dir / "timing-samples.csv", timings, references_raw,
                     supported, int(status.get("ntry", 3)), status.get("variants", {}))
    charts = build_charts(ducc, references, output_dir / "charts", supported)

    lines = ["# FFT inline-fix × FFT-tweaks benchmark report", ""]
    lines.extend(["## A. Executive overview", ""])
    cpu_info = status.get("cpu_info", {})
    cpu_identity = None
    for row in timings:
        if row.get("record_type") == "ducc":
            cpu_identity = row.get("cpu_identity")
            if cpu_identity:
                break
    cpu_label = (cpu_identity or {}).get("model") or status.get("runner_cpu_model") or "unavailable"
    profiles_text = ", ".join(supported) if supported else "none"
    skipped_text = ", ".join(skipped) if skipped else "none"
    expected = status.get("expected_variant_profile_case_cells", 0)
    completed = status.get("completed_variant_profile_case_cells", 0)
    failed = status.get("failed_correctness_cells", 0)
    incomplete = status.get("incomplete_variant_profile_case_cells", 0)
    compiler = status.get("compiler", {})
    lines.extend([
        f"- **Upstream base:** `mreineck/ducc:multiarch` at `{status.get('base_upstream_sha', manifest.get('base_sha', 'unavailable'))}`.",
        f"- **Benchmark source commit:** `{status.get('benchmark_source_head_sha', 'unavailable')}`.",
        f"- **Runner CPU:** {cpu_label}; architecture `{status.get('runner_architecture', 'unavailable')}`.",
        f"- **Compiler:** {compiler.get('version', 'unavailable')}.",
        f"- **Native comparison:** {native_status.get('status', 'unavailable')} (requested {native_status.get('requested_native_profile', 'unavailable')}, selected {native_profile or 'none'}); eight single-ISA source/LTO configurations.",
        f"- **Supported profiles:** {profiles_text}; **skipped:** {skipped_text}.",
        f"- **Benchmark cases:** {len(CASES)} operation/precision/dimension or fixed-shape cases per supported profile.",
        f"- **Factorial cells:** {expected} expected, {completed} fully timed, {failed} with correctness failures, {incomplete} incomplete.",
        f"- **LTO:** disabled for all four variants; upstream multiarch profile objects are linked directly.",
        f"- **Static ISA checks:** "
        f"{'passed for all four configurations' if len(status.get('variants', {})) == 4 and all(v.get('isa_validation') == 'pass' for v in status.get('variants', {}).values()) else 'incomplete or failed; see build diagnostics'}; these are not a comprehensive runtime ISA proof.",
        f"- **Run status:** `{status.get('overall_status', 'unavailable')}`.",
        f"- **Build workload:** `ntry={status.get('ntry', 3)}`, `nrepeat={status.get('nrepeat', 5)}`, one thread; references are timed once per profile/case/sample and reused across A–D.",
        f"- **Elapsed time:** source preparation {cell(status.get('source_preparation_seconds'), 1)} s; four builds and audits {cell(status.get('build_loop_seconds'), 1)} s; benchmark {cell(status.get('benchmark_seconds'), 1)} s; total {cell(status.get('total_run_seconds'), 1)} s.",
        "",
        "Configuration IDs used throughout:",
        "",
    ])
    lines.extend(markdown_table(
        ["ID", "special_mul inline fix", "FFT tweaks", "Label"],
        [[c["id"], "ON" if c["special_mul_fix"] else "OFF",
          "ON" if c["fft_tweaks"] else "OFF", c["label"]]
         for c in CONFIGS], align_right={0}))
    lines.append("")

    lines.extend(["## B. Complete absolute timing tables", "",
                  f"Values are milliseconds: for each sample, take the median of its repeated timed calls, then report the median across {status.get('ntry', 3)} deterministic samples. `—` means the cell has no complete timing record; the failures section identifies why.", ""])
    for profile in supported:
        lines.append(f"### {profile}")
        lines.append("")
        lines.append("Columns A–D use the configuration definitions immediately above this section.")
        lines.append("")
        lines.extend(markdown_table(["Case", *VARIANT_IDS], absolute_rows(ducc, profile)))
        lines.append("")
    if not supported:
        lines.append("No ISA profile was available to execute on this runner.")
        lines.append("")

    lines.extend(["## C. special_mul inline-fix effects", "",
                  "Each cell is paired `pre-inline-fix time / post-inline-fix time`; `>1` means the fix is faster. The two columns isolate the inline change with and without FFT tweaks.", ""])
    for profile in supported:
        add_case_table(lines, profile,
                       ["Case", "Without FFT tweaks (A/B)", "With FFT tweaks (C/D)"],
                       effect_rows(ducc, profile, "inline"))

    lines.extend(["## D. FFT-tweaks effects", "",
                  "Each cell is paired `without-tweaks time / with-tweaks time`; `>1` means the tweaks are faster. The two columns condition on the inline fix.", ""])
    for profile in supported:
        add_case_table(lines, profile,
                       ["Case", "Inline fix OFF (A/C)", "Inline fix ON (B/D)"],
                       effect_rows(ducc, profile, "tweaks"))

    lines.extend(["## E. Interaction and fixed regression probes", "",
                  "Interaction is `(C/D)/(A/B)`, computed on matched samples; `>1` means FFT tweaks amplify the inline-fix speedup. Combined effect is `A/D`; `>1` means the combined build is faster than the untreated baseline.", ""])
    fixed = [case for case in CASES if case["fixed_shape"] is not None]
    for profile in supported:
        add_case_table(lines, f"{profile} inline/tweaks interaction",
                       ["Case", "Inline fix, no tweaks (A/B)",
                        "Inline fix, tweaks ON (C/D)",
                        "Tweaks × inline interaction", "Combined gain (A/D)"],
                       interaction_rows(ducc, profile))
        add_case_table(lines, f"{profile} fixed complex128 c2c controls",
                       ["Fixed shape (not rounded)", *VARIANT_IDS],
                       absolute_rows(ducc, profile, fixed),
                       "Fixed 4095/4096 1D and both 2D orientations, without good_size rounding.")

    for reference, section in (("fftw", "F"), ("scipy", "G"), ("numpy", "H")):
        lines.extend([f"## {section}. {reference.upper()} comparisons", "",
                      f"Cells show `{reference.upper()} time / DUCC time`, paired by deterministic sample and shape; `>1` means DUCC is faster. Values are medians of per-sample ratios, not ratios of aggregate medians. The ISA label identifies the matched DUCC profile cap; reference packages use their normal installed builds and are not ISA-matched.", ""])
        for profile in supported:
            rows = []
            for case in CASES:
                rows.append([case["id"], *[
                    ratio_cell(reference_ratio(ducc, references, profile,
                                               case["id"], variant, reference))
                    for variant in VARIANT_IDS]])
            add_case_table(lines, profile, ["Case", *VARIANT_IDS], rows)
        if reference == "numpy":
            lines.extend(["### Actual output dtypes", "",
                          "The dtype table reports the actual output dtype returned by each installed library for every operation and input precision. Do not assume dtype promotion or preservation without checking these observations.", ""])
            lines.extend(markdown_table(
                ["Operation / input precision", "DUCC output", "FFTW output",
                 "SciPy output", "NumPy output"],
                dtype_rows(timings, references_raw)))
            lines.append("")

    lines.extend(["## I. ISA scaling", "",
                  "Cells use paired direct DUCC ratios: `v1 time / v3 time`, `v1 time / v4 time`, and `v3 time / v4 time`; `>1` means the higher profile is faster. A comparison is available only when both profiles ran on this same runner with matching shapes and input hashes.", ""])
    for upper, lower, title in (("x86-64-v3", "x86-64", "v3 vs v1"),
                                ("x86-64-v4", "x86-64", "v4 vs v1"),
                                ("x86-64-v4", "x86-64-v3", "v4 vs v3")):
        headers = ["Case", *VARIANT_IDS]
        rows = []
        if upper in supported and lower in supported:
            rows = [[case["id"], *[
                ratio_cell(scaling_ratio(ducc, lower, upper, case["id"], variant))
                for variant in VARIANT_IDS]] for case in CASES]
        else:
            rows = [[case["id"], *["unavailable" for _ in VARIANT_IDS]] for case in CASES]
        add_case_table(lines, f"{title}: {upper} / {lower}", headers, rows,
                       "The entire comparison is marked unavailable because at least one profile was unsupported on this runner." if not (upper in supported and lower in supported) else "")

    lines.extend(["## J. Build diagnostics", "",
                  "All four builds use the upstream direct multiarch profile-object link with LTO disabled. The temporary CMake worktree edit only turns global IPO off; it does not alter the committed upstream build.", ""])
    lines.append(f"FFT tweak patch SHA-256: `{manifest.get('fft_tweaks_provenance', {}).get('source_patch_sha256', 'unavailable')}`. Algorithm differences are limited to `fft1d_impl.h` and `fftnd_impl.h`; the inline factor changes only the `detail_fft::special_mul` annotation.")
    lines.append("")
    build_rows = []
    for config in CONFIGS:
        state = status.get("variants", {}).get(config["id"], {})
        build = state.get("build_diagnostics", {})
        build_rows.append([
            config["id"], state.get("build", "unavailable"),
            f"{build.get('total_build_seconds', 0):.1f}" if build.get("total_build_seconds") else "—",
            str(build.get("extension_size_bytes", "—")),
            str(build.get("warning_count", "—")),
            state.get("build_isolation", "unavailable"),
            state.get("isa_validation", "unavailable"),
        ])
    lines.extend(markdown_table(
        ["ID", "Build", "Build seconds", "Extension bytes",
         "Compiler warnings", "No-LTO link checks", "Static ISA checks"], build_rows))
    lines.append("")
    for config in CONFIGS:
        diag = build_diagnostics.get(config["id"], {})
        if diag.get("command_validation"):
            profiles = diag.get("profile_compile_validation", {})
            lines.append(f"- **{config['id']} build:** direct profile objects; " +
                         ", ".join(f"{p}: `-march={v['march']}`, {v['compile_commands']} TUs"
                                   for p, v in profiles.items()) +
                         "; no LTO in compile or link flags.")
            lines.append(f"  Final link: `{diag.get('final_link_command', 'unavailable')}`")
        if diag.get("warning_lines"):
            lines.append(f"- **{config['id']} warnings:** " +
                         "; ".join(item.replace("`", "'")
                                   for item in diag["warning_lines"][:3]))
    lines.extend(["", "### Static ISA validation", "",
                  "The inspection checks representative FFT profile objects, baseline dispatcher code, and the final extension for LTO sections. Passing does not establish comprehensive ISA purity across all native symbols or every supported CPU.", ""])
    isa_rows = []
    for config in CONFIGS:
        note = isa_notes.get(config["id"], {})
        profiles = note.get("profile_disassembly", {})
        isa_rows.append([
            config["id"], note.get("status", "unavailable"),
            ", ".join(f"{p}: {v.get('disassembly', {}).get('instruction_count', 0)} instructions"
                      for p, v in profiles.items()) or "—",
            "checked" if note.get("dispatcher_disassembly") else "unavailable",
            "no LTO IR" if note.get("extension_gnu_lto_sections") is False else "unavailable",
        ])
    lines.extend(markdown_table(
        ["ID", "Static check", "Representative FFT objects",
         "Dispatcher", "Extension"], isa_rows))
    lines.append("")

    lines.extend(["## K. Failures and limitations", ""])
    profile_rows = []
    for profile in PROFILES:
        if profile in supported:
            profile_rows.append([profile, "measured", "available in upstream cpu_info()"])
        else:
            profile_rows.append([profile, "skipped", "not reported as available by upstream cpu_info() on this runner"])
    lines.extend(markdown_table(["ISA profile", "Status", "Reason"], profile_rows,
                                align_right=set()))
    lines.append("")

    failures = []
    for config in CONFIGS:
        state = status.get("variants", {}).get(config["id"], {})
        if state.get("build") != "pass":
            failures.append(["build", config["id"], "all supported profiles", "all cases",
                             state.get("build_error") or state.get("source_error") or
                             state.get("build", "not run")])
        if state.get("build_isolation") != "pass" or state.get("isa_validation") != "pass":
            if state.get("validation_error"):
                failures.append(["build/ISA validation", config["id"], "all supported profiles",
                                 "all cases", state["validation_error"]])
        if state.get("runtime_import") != "pass":
            failures.append(["runtime import", config["id"], "all supported profiles",
                             "all cases", state.get("runtime_import_error", "not run")])

    for row in timings:
        if row.get("record_type") == "ducc" and row.get("correctness") != "pass":
            failures.append([
                "correctness", row.get("variant", "?"), row.get("profile", "?"),
                f"{row.get('operation')} {row.get('case')} {row.get('precision')} shape={row.get('shape')}",
                row.get("correctness_error") or row.get("correctness", "unavailable"),
            ])
    for row in references_raw:
        if row.get("correctness") != "pass":
            failures.append([
                "reference correctness", row.get("reference", "?"), row.get("profile", "?"),
                f"{row.get('operation')} {row.get('case')} {row.get('precision')} shape={row.get('shape')}",
                row.get("correctness_error") or row.get("correctness", "unavailable"),
            ])

    expected_keys = [(profile, case["id"], variant)
                     for profile in supported for case in CASES for variant in VARIANT_IDS]
    for profile, case_id, variant in expected_keys:
        rows = _rows_for(ducc, profile, case_id, variant)
        if len(rows) != int(status.get("ntry", 3)):
            failures.append(["incomplete timing", variant, profile, case_id,
                             f"{len(rows)} of {status.get('ntry', 3)} sample records"])
    failures.extend(["benchmark worker", item.get("variant", "?"),
                     item.get("profile", "?"), item.get("case", "?"),
                     item.get("error", item.get("correctness_error", "worker error"))]
                    for item in status.get("benchmark", {}).get("worker_failures", []))
    if status.get("benchmark_error"):
        failures.append(["benchmark run", "all", "all supported profiles", "all cases",
                         status["benchmark_error"]])
    if failures:
        lines.extend(markdown_table(
            ["Failure type", "Variant / reference", "ISA", "Operation / case / shape", "Detail"],
            [[str(value) for value in row] for row in failures], align_right=set()))
    else:
        lines.append("No build, correctness, timing-completeness, or worker failures were recorded.")
    lines.append("")

    lines.extend(["### Unavailable comparisons", ""])
    unavailable = []
    for profile in supported:
        for case in CASES:
            for variant in VARIANT_IDS:
                if ducc_absolute(ducc, profile, case["id"], variant) is None:
                    unavailable.append([profile, case["id"], variant,
                                        "incorrect result or incomplete DUCC timing samples"])
                for reference in DISPLAY_REFERENCES:
                    if reference_ratio(ducc, references, profile, case["id"],
                                       variant, reference) is None:
                        unavailable.append([profile, case["id"], variant,
                                            f"{reference.upper()} / DUCC comparison unavailable"])
    if unavailable:
        lines.extend(markdown_table(["ISA", "Case", "Variant", "Unavailable comparison"],
                                    unavailable, align_right=set()))
    else:
        lines.append("All DUCC absolute timings and all reference comparisons for supported profiles are present.")
    lines.extend(["## L. Single-ISA native × LTO × inline/tweaks matrix", "",
                  "The native builds contain only the selected ISA, with either complete same-ISA LTO or no LTO. They do **not** use the multiarch profile-local LTO strategy. All native and multiarch timings are from the same hosted runner, but measurements occurred sequentially and remain sensitive to runner noise.", ""])
    native_cpu = native_status.get("cpu_identity", {})
    native_features = set(native_cpu.get("features", []))
    cpu_flags = ", ".join(flag for flag in (
        "sse2", "sse4_2", "avx", "avx2", "fma",
        "avx512f", "avx512dq", "avx512bw", "avx512vl")
        if flag in native_features) or "unavailable"
    lines.extend([
        f"- **Host CPU model:** {native_cpu.get('model') or cpu_label}.",
        f"- **Host OS:** {native_status.get('host_os', 'unavailable')}.",
        f"- **CPU feature flags:** {cpu_flags}.",
        f"- **Reported available multiarch profiles:** {', '.join(native_status.get('available_profiles', [])) or 'unavailable'}.",
        f"- **Requested native ISA:** {native_status.get('requested_native_profile', 'unavailable')}; **selected:** {native_profile or 'none'}.",
        f"- **Native validation:** {native_status.get('status', 'unavailable')}; eight builds expected, {len(native_ducc)} native timing samples indexed (by profile/case/variant/sample).",
        f"- **Native compiler:** {native_status.get('compiler', {}).get('version', 'unavailable')}.",
        "",
        "Native source configurations retain A–D: pre-inline/no-tweaks, inline/no-tweaks, pre-inline/tweaks, and inline/tweaks. Native IDs `NA0`–`ND0` use LTO OFF; `NA1`–`ND1` use LTO ON. **Neither LTO-ON native build nor its results establish multiarch LTO safety.**",
        "",
    ])
    if native_profile and native_status.get("status") != "skipped":
        native_build_rows = []
        for lto in (0, 1):
            for variant in VARIANT_IDS:
                label = f"N{variant}{lto}"
                entry = native_status.get("variants", {}).get(label, {})
                build = entry.get("build_diagnostics", {})
                native_build_rows.append([
                    label, native_profile, "ON" if lto else "OFF",
                    "ON" if entry.get("inline") else "OFF",
                    "ON" if entry.get("tweaks") else "OFF",
                    entry.get("build", "unavailable"),
                    cell(sum(build.get("build_times", {}).values()), 1)
                    if build.get("build_times") else "—",
                    str(build.get("size_bytes", "—")),
                    entry.get("error", "—")[:300],
                ])
        add_case_table(lines, "Native build/flag validation",
                       ["ID", "Exact ISA", "LTO", "Inline", "Tweaks", "Build", "Build seconds",
                        "Extension bytes", "Error"], native_build_rows,
                       "Every successful build verifies -march, IPO compile/link flags, a native Python import, and absence of LTO IR in the final extension.")
        add_case_table(lines, f"Native absolute execution time: {native_profile}",
                       ["Case", *native_ids],
                       [[case["id"], *[absolute_cell(native_ducc, native_profile,
                                                    case["id"], variant)
                                       for variant in native_ids]] for case in CASES],
                       "Milliseconds, median of repeat medians across matched deterministic samples.")
        add_case_table(lines, "Native LTO effect (OFF time / ON time)",
                       ["Case", *[f"{v}: N{v}0/N{v}1" for v in VARIANT_IDS]],
                       [[case["id"], *[
                           ratio_cell(factor_ratio(native_ducc, native_profile,
                                                   case["id"], f"N{v}0", f"N{v}1"))
                           for v in VARIANT_IDS]] for case in CASES],
                       "Values above 1.0 indicate faster LTO-ON code at the same ISA, inline setting, tweaks setting, and input.")
        add_case_table(lines, "Native inline and FFT-tweak effects",
                       ["Case", "Inline no tweaks LTO OFF", "Inline tweaks LTO OFF",
                        "Tweaks inline ON LTO OFF", "Inline no tweaks LTO ON",
                        "Inline tweaks LTO ON", "Tweaks inline ON LTO ON"],
                       [[case["id"], *[
                           ratio_cell(factor_ratio(native_ducc, native_profile,
                                                   case["id"], left, right))
                           for left, right in (
                               ("NA0","NB0"), ("NC0","ND0"), ("NB0","ND0"),
                               ("NA1","NB1"), ("NC1","ND1"), ("NB1","ND1"))
                           ]] for case in CASES],
                       "OFF configuration time / ON configuration time; above 1.0 favors the latter.")
        if native_profile in supported:
            add_case_table(lines, "Native versus matching multiarch ISA",
                           ["Case", *native_ids],
                           [[case["id"], *[
                               ratio_cell(factor_ratio(combined_ducc, native_profile,
                                                       case["id"], variant[1], variant))
                               for variant in native_ids]] for case in CASES],
                           "Multiarch time / native time at the same named ISA, source configuration and input. Above 1.0 favors native. The multiarch extension has global IPO disabled; native LTO-ON differs intentionally.")
        for reference in DISPLAY_REFERENCES:
            add_case_table(lines, f"{reference.upper()} / native ({native_profile})",
                           ["Case", *native_ids],
                           [[case["id"], *[
                               ratio_cell(reference_ratio(native_ducc, native_references,
                                                          native_profile, case["id"],
                                                          variant, reference))
                               for variant in native_ids]] for case in CASES],
                           "Reference time / native DUCC time; above 1.0 means native DUCC is faster. The reference wheel is not ISA-matched.")
        native_benchmark = native_status.get("benchmark", {})
        lines.extend([
            f"Native timing samples: **{native_benchmark.get('completed_samples', 0)} / {native_benchmark.get('expected_samples', 0)}**; correctness failures: {native_benchmark.get('correctness_failures', 'unavailable')}.",
            "",
        ])
        if native_benchmark.get("errors"):
            lines.append("Native benchmark errors:")
            for error in native_benchmark["errors"][:50]:
                lines.append(f"- {str(error)[:400]}")
            lines.append("")
        failed_builds = [v for v, item in native_status.get("variants", {}).items()
                         if item.get("build") != "pass"]
        if failed_builds:
            lines.append(f"Native build failures: {', '.join(failed_builds)}.")
            lines.append("")
    else:
        lines.extend([f"Native matrix not executed: {native_status.get('error') or 'native-status.json missing or native ISA unavailable'}.", ""])
    lines.extend(["Native raw timing and reference records are preserved in `native-timings.jsonl` and `native-references.jsonl`; compiler/link diagnostics in `native-status.json` and `native-build-logs/`.", ""])

    lines.extend(["", f"**Primary PNG charts:** {len(charts)} (maximum 9).", ""])
    lines.append("Charts are stored in the downloadable `fft-inline-tweaks-results` Actions artifact; relative artifact paths cannot render as images inside an Actions step summary.")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repository and run_id:
        lines.append(f"[View this run and its downloadable artifact](https://github.com/{repository}/actions/runs/{run_id})")
    for chart in charts:
        lines.append(f"- `{chart}`")
    lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report, len(charts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path,
                        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
                        if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, chart_count = make_report(output_dir)
    report_bytes = report.encode("utf-8")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        if len(report_bytes) > MAX_SUMMARY_BYTES:
            args.summary.write_text(
                f"# FFT inline-fix × FFT-tweaks benchmark\n\nThe complete report is "
                f"{len(report_bytes)} bytes, above the {MAX_SUMMARY_BYTES}-byte "
                "Actions step-summary limit. It is preserved as `report.md` in the "
                "artifact; the workflow is marked failed so no report data is silently "
                "truncated.\n", encoding="utf-8")
            return 2
        args.summary.write_bytes(report_bytes)
    print(f"wrote report.md ({len(report_bytes)} bytes) and {chart_count} PNG charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
