#!/usr/bin/env python3
"""Render complete Markdown, CSV, and compact reference/DUCC heatmaps."""

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
    rows = []
    for case in CASES:
        values = []
        if factor == "lto":
            combos = [(inline, tweaks) for inline in (False, True)
                      for tweaks in (False, True)]
            for inline, tweaks in combos:
                off = next(c["id"] for c in CONFIGS if not c["profile_lto"]
                           and c["special_mul_fix"] == inline and c["fft_tweaks"] == tweaks)
                on = next(c["id"] for c in CONFIGS if c["profile_lto"]
                          and c["special_mul_fix"] == inline and c["fft_tweaks"] == tweaks)
                values.append(ratio_cell(factor_ratio(ducc, profile, case["id"], off, on)))
        elif factor == "inline":
            combos = [(lto, tweaks) for lto in (False, True)
                      for tweaks in (False, True)]
            for lto, tweaks in combos:
                off = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                           and not c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
                on = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                          and c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
                values.append(ratio_cell(factor_ratio(ducc, profile, case["id"], off, on)))
        else:
            combos = [(lto, inline) for lto in (False, True)
                      for inline in (False, True)]
            for lto, inline in combos:
                off = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                           and c["special_mul_fix"] == inline and not c["fft_tweaks"])
                on = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                          and c["special_mul_fix"] == inline and c["fft_tweaks"])
                values.append(ratio_cell(factor_ratio(ducc, profile, case["id"], off, on)))
        rows.append([case["id"], *values])
    return rows


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
    rows = []
    for case in CASES:
        cid = case["id"]
        values = []
        # Inline ON/OFF changes the LTO speedup; show both FFT-tweak states.
        for tweaks in (False, True):
            nofix_no_lto = next(c["id"] for c in CONFIGS if not c["profile_lto"]
                                and not c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
            nofix_lto = next(c["id"] for c in CONFIGS if c["profile_lto"]
                             and not c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
            fix_no_lto = next(c["id"] for c in CONFIGS if not c["profile_lto"]
                              and c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
            fix_lto = next(c["id"] for c in CONFIGS if c["profile_lto"]
                           and c["special_mul_fix"] and c["fft_tweaks"] == tweaks)
            values.append(ratio_cell(_paired_ratio_of_factor_ratios(
                ducc, profile, cid, fix_no_lto, fix_lto, nofix_no_lto, nofix_lto)))

        # FFT tweaks change the LTO speedup; show both inline-fix states.
        for inline in (False, True):
            no_tweak_no_lto = next(c["id"] for c in CONFIGS if not c["profile_lto"]
                                   and c["special_mul_fix"] == inline and not c["fft_tweaks"])
            no_tweak_lto = next(c["id"] for c in CONFIGS if c["profile_lto"]
                                and c["special_mul_fix"] == inline and not c["fft_tweaks"])
            tweak_no_lto = next(c["id"] for c in CONFIGS if not c["profile_lto"]
                                and c["special_mul_fix"] == inline and c["fft_tweaks"])
            tweak_lto = next(c["id"] for c in CONFIGS if c["profile_lto"]
                             and c["special_mul_fix"] == inline and c["fft_tweaks"])
            values.append(ratio_cell(_paired_ratio_of_factor_ratios(
                ducc, profile, cid, tweak_no_lto, tweak_lto,
                no_tweak_no_lto, no_tweak_lto)))

        # FFT tweaks change the inline-fix effect; show both LTO states.
        for lto in (False, True):
            no_tweak_pre = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                                and not c["special_mul_fix"] and not c["fft_tweaks"])
            no_tweak_fix = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                                and c["special_mul_fix"] and not c["fft_tweaks"])
            tweak_pre = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                             and not c["special_mul_fix"] and c["fft_tweaks"])
            tweak_fix = next(c["id"] for c in CONFIGS if c["profile_lto"] == lto
                             and c["special_mul_fix"] and c["fft_tweaks"])
            values.append(ratio_cell(_paired_ratio_of_factor_ratios(
                ducc, profile, cid, tweak_pre, tweak_fix, no_tweak_pre, no_tweak_fix)))

        # Compare H/A with a multiplicative prediction from the three A-only effects.
        a_to_e = _paired_samples(_rows_for(ducc, profile, cid, "A"),
                                 _rows_for(ducc, profile, cid, "E"))
        a_to_b = _paired_samples(_rows_for(ducc, profile, cid, "A"),
                                 _rows_for(ducc, profile, cid, "B"))
        a_to_c = _paired_samples(_rows_for(ducc, profile, cid, "A"),
                                 _rows_for(ducc, profile, cid, "C"))
        a_to_h = _paired_samples(_rows_for(ducc, profile, cid, "A"),
                                 _rows_for(ducc, profile, cid, "H"))
        triple = None
        if all(items is not None for items in (a_to_e, a_to_b, a_to_c, a_to_h)):
            if len({len(items) for items in (a_to_e, a_to_b, a_to_c, a_to_h)}) == 1:
                triple = median_or_none([
                    observed / (lto * inline * tweaks)
                    for observed, lto, inline, tweaks in zip(a_to_h, a_to_b, a_to_e, a_to_c)
                    if min(observed, lto, inline, tweaks) > 0
                ])
        values.append(ratio_cell(triple))
        rows.append([cid, *values])
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
    supported = [profile for profile in PROFILES
                 if profile in status.get("supported_profiles", [])]
    skipped = [profile for profile in PROFILES if profile not in supported]

    write_timing_csv(output_dir / "timing-samples.csv", timings, references_raw,
                     supported, int(status.get("ntry", 3)), status.get("variants", {}))
    charts = build_charts(ducc, references, output_dir / "charts", supported)

    lines = ["# FFT factorial benchmark report", ""]
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
        f"- **Supported profiles:** {profiles_text}; **skipped:** {skipped_text}.",
        f"- **Benchmark cases:** {len(CASES)} operation/precision/dimension or fixed-shape cases per supported profile.",
        f"- **Factorial cells:** {expected} expected, {completed} fully timed, {failed} with correctness failures, {incomplete} incomplete.",
        f"- **LTO implementation:** profile-local LTO with three independent native `nolto-rel` partial links and a non-LTO final extension link; no combined cross-profile LTO.",
        f"- **ISA isolation:** "
        f"{'passed for every buildable configuration' if all(v.get('isa_validation') == 'pass' for v in status.get('variants', {}).values()) else 'incomplete or failed; see build diagnostics'}.",
        f"- **Run status:** `{status.get('overall_status', 'unavailable')}`.",
        f"- **Build workload:** `ntry={status.get('ntry', 3)}`, `nrepeat={status.get('nrepeat', 5)}`, one thread; references are timed once per profile/case/sample and reused across A–H.",
        f"- **Elapsed time:** source preparation {cell(status.get('source_preparation_seconds'), 1)} s; eight builds and audits {cell(status.get('build_loop_seconds'), 1)} s; benchmark {cell(status.get('benchmark_seconds'), 1)} s; total {cell(status.get('total_run_seconds'), 1)} s.",
        "",
        "Configuration IDs used throughout:",
        "",
    ])
    lines.extend(markdown_table(
        ["ID", "Profile LTO", "special_mul inline fix", "FFT tweaks", "Label"],
        [[c["id"], "ON" if c["profile_lto"] else "OFF",
          "ON" if c["special_mul_fix"] else "OFF",
          "ON" if c["fft_tweaks"] else "OFF", c["label"]]
         for c in CONFIGS], align_right={0}))
    lines.append("")

    lines.extend(["## B. Complete absolute timing tables", "",
                  f"Values are milliseconds: for each sample, take the median of its repeated timed calls, then report the median across {status.get('ntry', 3)} deterministic samples. `—` means the cell has no complete timing record; the failures section identifies why.", ""])
    for profile in supported:
        lines.append(f"### {profile}")
        lines.append("")
        lines.append("Columns A–H use the configuration definitions immediately above this section.")
        lines.append("")
        lines.extend(markdown_table(["Case", *VARIANT_IDS], absolute_rows(ducc, profile)))
        lines.append("")
    if not supported:
        lines.append("No ISA profile was available to execute on this runner.")
        lines.append("")

    lines.extend(["## C. LTO effect tables", "",
                  "Each cell is the paired per-sample ratio `LTO OFF time / profile-local LTO time`; `>1` means profile-local LTO is faster. Columns condition on the inline-fix and FFT-tweaks states.", ""])
    for profile in supported:
        headers = ["Case", "Inline OFF, tweaks OFF", "Inline ON, tweaks OFF",
                   "Inline OFF, tweaks ON", "Inline ON, tweaks ON"]
        add_case_table(lines, profile, headers, effect_rows(ducc, profile, "lto"))

    lines.extend(["## D. special_mul inline-fix effect tables", "",
                  "Each cell is `pre-inline-fix time / post-inline-fix time`; `>1` means retaining the upstream always-inline fix is faster. Columns condition on profile LTO and FFT tweaks.", ""])
    for profile in supported:
        headers = ["Case", "LTO OFF, tweaks OFF", "LTO OFF, tweaks ON",
                   "LTO ON, tweaks OFF", "LTO ON, tweaks ON"]
        add_case_table(lines, profile, headers, effect_rows(ducc, profile, "inline"))

    lines.extend(["## E. FFT-tweaks effect tables", "",
                  "Each cell is `without-tweaks time / with-tweaks time`; `>1` means the FFT tweaks are faster. Columns condition on profile LTO and inline-fix state.", ""])
    for profile in supported:
        headers = ["Case", "LTO OFF, inline OFF", "LTO OFF, inline ON",
                   "LTO ON, inline OFF", "LTO ON, inline ON"]
        add_case_table(lines, profile, headers, effect_rows(ducc, profile, "tweaks"))

    lines.extend(["## F. Interactions and fixed regression probes", "",
                  "Interaction values are paired ratios of conditional factor ratios, calculated per sample before taking the median. A value above 1 means the first named state increases the second factor's speedup. The final column compares the observed H-versus-A speedup with a multiplicative no-interaction prediction from A's three one-factor effects.", ""])
    interaction_headers = [
        "Case", "Inline × LTO, tweaks OFF", "Inline × LTO, tweaks ON",
        "Tweaks × LTO, inline OFF", "Tweaks × LTO, inline ON",
        "Tweaks × inline, LTO OFF", "Tweaks × inline, LTO ON",
        "H observed / independent prediction",
    ]
    for profile in supported:
        add_case_table(lines, f"{profile} interactions", interaction_headers,
                       interaction_rows(ducc, profile),
                       "For the pairwise columns, `>1` means the first factor increases the second factor's speedup. For `H observed / independent prediction`, `>1` means H is faster than the prediction from the three A-only effects.")
        fixed = [case for case in CASES if case["fixed_shape"] is not None]
        add_case_table(lines, f"{profile} fixed complex128 c2c controls",
                       ["Fixed shape (never passed to good_size)", *VARIANT_IDS],
                       absolute_rows(ducc, profile, fixed),
                       "The six fixed controls are 4095, 4096, (64,4095), (64,4096), (4095,64), and (4096,64).")

    for reference, section in (("fftw", "G"), ("scipy", "H"), ("numpy", "I")):
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
                          "The dtype table reports observed output dtypes. NumPy's FFT API promotes single-precision inputs to double precision; FFTW, SciPy, and DUCC retain single-precision outputs where supported.", ""])
            lines.extend(markdown_table(
                ["Operation / input precision", "DUCC output", "FFTW output",
                 "SciPy output", "NumPy output"],
                dtype_rows(timings, references_raw)))
            lines.append("")

    lines.extend(["## J. ISA scaling", "",
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

    lines.extend(["## K. Build diagnostics", ""])
    lines.append(f"FFT tweak source patch SHA-256: `{manifest.get('fft_tweaks_provenance', {}).get('source_patch_sha256', 'unavailable')}`. The algorithm patch applies only to `fft1d_impl.h` and `fftnd_impl.h`; `DUCC0_NAMESPACE` remains defined by the pinned upstream source. The historical benchmark-only tweak commit is excluded.")
    lines.append("")
    lines.append(f"Benchmark CMake helper SHA-256: `{load_json(output_dir / 'patch-provenance.json', {}).get('profile_local_lto_helper_sha256', 'unavailable')}`. LTO ON builds compile each profile's translation units with its exact `-march` and LTO, partial-link only that profile using GCC `-r -flto -flto-partition=none -flinker-output=nolto-rel`, then link three native outputs with baseline dispatcher objects and `-fno-lto`.")
    lines.append("")
    lines.append("The baseline dispatcher contains one upstream `xgetbv` probe to read OS vector-state support. The audit confirms both CPUID XSAVE and OSXSAVE bits are checked before the probe; profile code contains no CPU-detection probe.")
    lines.append("")
    build_rows = []
    for config in CONFIGS:
        state = status.get("variants", {}).get(config["id"], {})
        build = state.get("build_diagnostics", {})
        iso = state.get("build_isolation", "unavailable")
        isa = state.get("isa_validation", "unavailable")
        build_rows.append([
            config["id"], state.get("build", "unavailable"),
            "profile-local" if config["profile_lto"] else "off",
            f"{build.get('total_build_seconds', 0):.1f}" if build.get("total_build_seconds") else "—",
            str(build.get("extension_size_bytes", "—")),
            str(build.get("warning_count", "—")), iso, isa,
        ])
    lines.extend(markdown_table(
        ["ID", "Build", "Actual LTO", "Build seconds", "Extension bytes",
         "Compiler warnings", "Link checks", "ISA checks"], build_rows))
    lines.append("")
    for config in CONFIGS:
        diag = build_diagnostics.get(config["id"], {})
        if diag.get("command_validation"):
            lines.append(f"- **{config['id']} commands:** `{diag['lto_implementation']}`; profile compile commands: " + ", ".join(
                f"{p} `{v['compile_commands']}` with `-march={v['march']}`, PIC and LTO `{v['lto']}`"
                for p, v in diag["profile_compile_validation"].items()) + ".")
            lines.append(f"  Final link: `{diag['final_link_command']}`")
        if diag.get("warning_lines"):
            lines.append(f"- **{config['id']} compiler warnings:** {diag.get('warning_count')} total; examples: `" + "` · `".join(
                item.replace("`", "'") for item in diag["warning_lines"][:3]) + "`.")
    if not any(build_diagnostics.values()):
        lines.append("Build commands and ISA checks are unavailable because no build diagnostics were written.")
    lines.append("")
    lines.extend(["### ISA validation summary", ""])
    isa_rows = []
    for config in CONFIGS:
        note = isa_notes.get(config["id"], {})
        profile_notes = note.get("profile_disassembly", {})
        counts = ", ".join(
            f"{profile}: {item.get('disassembly', {}).get('instruction_count', 0)} instructions"
            for profile, item in profile_notes.items())
        probes = note.get("guarded_cpu_feature_probes", [])
        probe_text = ", ".join(
            f"{item.get('instruction')} guarded ({item.get('count')})" for item in probes) or "none"
        isa_rows.append([config["id"], note.get("status", "unavailable"), counts or "—",
                         "baseline dispatcher checked" if note.get("dispatcher_disassembly") else "—",
                         probe_text,
                         "no LTO IR in final extension" if note.get("extension_gnu_lto_sections") is False else "—"])
    lines.extend(markdown_table(["ID", "Status", "Profile disassembly", "Dispatcher",
                                 "CPU feature probes", "Final link"], isa_rows))
    lines.append("")

    reverse = build_diagnostics.get("E", {}).get("reverse_link", {})
    reverse_state = status.get("variants", {}).get("E", {})
    lines.extend(["### Supplemental E reverse-order final link", ""])
    if reverse.get("reversed_profile_object_order"):
        correctness = reverse.get("correctness", {})
        reverse_rows = [
            ["Reverse check status", reverse.get("status", "unavailable")],
            ["Normal profile object order", ", ".join(reverse.get("normal_profile_object_order", []))],
            ["Reversed profile object order", ", ".join(reverse.get("reversed_profile_object_order", []))],
            ["Profile-local LTO partial links", f"{reverse.get('partial_link_command_count', 0)} existing links reused unchanged (SHA-256 `{reverse.get('partial_link_commands_sha256', 'unavailable')}`)" if reverse.get("profile_local_partial_links_reused_unchanged") else "validation failed"],
            ["Other link inputs", "same baseline dispatcher and remaining object/library inputs" if reverse.get("other_link_inputs_unchanged") else "validation failed"],
            ["Final link flags", "-fno-lto; no -flto flags" if reverse.get("final_link_fno_lto") and not reverse.get("final_link_lto_flags") else "validation failed"],
            ["Reversed extension", f"{reverse.get('extension_size_bytes', '—')} bytes at `{reverse.get('extension', 'unavailable')}`; no .gnu.lto sections" if reverse.get("gnu_lto_sections") is False else "validation failed"],
            ["Python initializer", reverse.get("python_init_symbol", "unavailable")],
            ["Profile namespaces", ", ".join(f"{profile}: {count}" for profile, count in reverse.get("profile_symbol_counts", {}).items())],
            ["v4 runtime", reverse.get("v4_execution", "unavailable")],
        ]
        lines.extend(markdown_table(["Check", "Result"], reverse_rows, align_right=set()))
        lines.append("")
        lines.append(f"The reversed extension was linked from the existing E native objects with the recorded baseline dispatcher and remaining link inputs: `{reverse.get('final_link_command', 'unavailable')}`. Correctness checks ran against the {len(CASES)} case set at v1 and v3 and stopped at the first process failure; v4 was only compiled and inspected.")
        lines.append("")
        correctness_rows = [[
            {"1": "x86-64", "3": "x86-64-v3"}.get(profile, profile),
            row.get("import", "unavailable"),
            str(row.get("case_count", 0)), row.get("correctness", "unavailable"),
            row.get("failed_or_interrupted_case", "—"),
            ("SIGILL (-4)" if row.get("process_returncode") == -4 else
             str(row.get("process_returncode", "—"))),
        ] for profile, row in correctness.items()]
        lines.extend(markdown_table(["Executed profile", "Import", "Cases completed", "Correctness", "Failed/interrupted case", "Process status"],
                                    correctness_rows, align_right={2, 5}))
        lines.append("")
        disassembly_rows = [[
            label, details.get("symbol", "unavailable"),
            str(details.get("instruction_count", "—")),
            "match" if details.get("normalized_instruction_stream_match") else "mismatch",
            str(details.get("normalized_instruction_change_count", 0)),
            "match" if details.get("raw_instruction_encoding_match") else "relocations differ",
            str(details.get("rip_relative_reference_changes", 0)),
            str(details.get("rip_relative_symbol_changes", 0)),
            f"{details.get('reversed_vex_instruction_count', 0)} VEX / {details.get('reversed_evex_instruction_count', 0)} EVEX",
        ] for label, details in reverse.get("disassembly_comparison", {}).items()]
        lines.extend(markdown_table(["Disassembled code", "Symbol", "Instructions", "Instruction stream", "Changed instructions", "Encoding", "RIP targets changed", "Target symbols changed", "Reversed encodings"],
                                    disassembly_rows, align_right={2, 4, 6, 7}))
        lines.append("")
        lines.append("Per-symbol disassembly covers one FFT execution routine from each profile, CPU capability detection, `PyInit_ducc0`, and the shared `std::vector<unsigned long>` copy constructor. The instruction-stream comparison normalizes relocated branch and RIP-relative addresses; the table separately counts changed instructions, resolved targets, base symbol changes, raw encodings, and VEX/EVEX instructions. Full target examples, commands, hashes, and disassembly are in the E build log, `reverse-link-validation.json`, and `build-diagnostics.json`.")
        if reverse.get("detected_link_order_instruction_changes"):
            lines.append("Detected instruction-stream changes: " + ", ".join(
                f"`{label}` ({reverse['disassembly_comparison'][label].get('normalized_instruction_change_count')} changed instruction positions; reversed EVEX count {reverse['disassembly_comparison'][label].get('reversed_evex_instruction_count')})"
                for label in reverse["detected_link_order_instruction_changes"]) + ".")
            lines.append("")
            change_rows = []
            for label in reverse["detected_link_order_instruction_changes"]:
                for item in reverse["disassembly_comparison"][label].get(
                        "normalized_instruction_change_examples", [])[:6]:
                    change_rows.append([
                        label, str(item.get("instruction_index")),
                        item.get("normal", "—"), item.get("reversed", "—"),
                    ])
            lines.extend(markdown_table(
                ["Changed symbol", "Instruction index", "Normal order", "Reversed order"],
                change_rows, align_right={1}))
            lines.append("")
        if reverse.get("error"):
            lines.append(f"Reverse-link finding: {reverse['error']}")
            lines.append("")
    else:
        lines.append(f"Reverse-order final-link validation: **{reverse_state.get('reverse_link_validation', 'unavailable')}**. {reverse.get('error') or reverse_state.get('reverse_link_error', 'No reverse-link result was recorded.')}")
    lines.append("")

    lines.extend(["## L. Failures and limitations", ""])
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
        if config["id"] == "E" and state.get("reverse_link_validation") != "pass":
            failures.append(["reverse-order link", "E", "v1 and v3 correctness",
                             "24-case set, v1/v3 only",
                             state.get("reverse_link_error", "not run")])

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
    lines.extend(["", f"**Primary PNG charts:** {len(charts)} (maximum 9).", ""])
    for chart in charts:
        lines.append(f"![{Path(chart).stem}]({chart})")
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
                f"# FFT factorial benchmark\n\nThe complete report is "
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
