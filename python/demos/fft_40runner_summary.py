#!/usr/bin/env python3
"""Summarize paired FFT benchmark rows by physical CPU and precision."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


COMPARISONS = (
    ("candidate-v3", "upstream-v3", "v3 / upstream v3"),
    ("candidate-v4", "upstream-v4", "v4 / upstream v4"),
    ("candidate-v3", "upstream-v1", "v3 / upstream v1"),
    ("candidate-v4", "upstream-v1", "v4 / upstream v1"),
    ("candidate-v4", "candidate-v3", "v4 / candidate v3"),
)
SUITES = ("1d", "batched", "strided", "nd", "long",
          "real", "r2r", "convolution")


def median(values):
    return statistics.median(values)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = (len(ordered) - 1) * fraction
    low = int(idx)
    high = min(len(ordered) - 1, low + 1)
    if high == low:
        return ordered[low]
    return ordered[low] * (high - idx) + ordered[high] * (idx - low)


def load_rows(root):
    records = {}
    coverage = defaultdict(set)
    issues = []
    for file in root.rglob("rows.jsonl"):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["shard"], row["case_id"], row["profile"])
            if key in records:
                issues.append(f"duplicate row: {key}")
                continue
            records[key] = row
            coverage[row["shard"]].add(row["profile"])
            if row["status"] != "ok":
                issues.append(f"invalid benchmark: {key}: {row.get('error')}")
    return records, coverage, issues


def paired_rows(rows):
    pairs = []
    for (shard, case_id, profile), candidate in rows.items():
        if candidate["status"] != "ok":
            continue
        for new_profile, reference_profile, label in COMPARISONS:
            if profile != new_profile:
                continue
            ref = rows.get((shard, case_id, reference_profile))
            if ref is None or ref["status"] != "ok":
                continue
            if candidate["cpu_model"] != ref["cpu_model"]:
                raise ValueError("CPU model changed within runner")
            if candidate["cpu_isa"] != ref["cpu_isa"]:
                raise ValueError("CPU ISA changed within runner")
            ratio = candidate["ns_median"] / ref["ns_median"]
            pairs.append(dict(
                shard=shard,
                case_id=case_id,
                cpu_model=candidate["cpu_model"],
                cpu_isa=candidate["cpu_isa"],
                precision=candidate["precision"],
                dtype=candidate["dtype"],
                suite=candidate["suite"],
                kind=candidate["kind"],
                shape=candidate["shape"],
                axes=candidate["axes"],
                layout=candidate["layout"],
                threads=candidate["threads"],
                forward=candidate["forward"],
                inplace=candidate["inplace"],
                transform_type=candidate["transform_type"],
                comparison=label,
                candidate_profile=new_profile,
                reference_profile=reference_profile,
                candidate_ns=candidate["ns_median"],
                reference_ns=ref["ns_median"],
                regression_percent=(ratio - 1) * 100.0,
                speedup=1.0 / ratio,
            ))
    return pairs


def consolidate(pairs):
    by_case = defaultdict(list)
    for row in pairs:
        key = (row["cpu_model"], row["cpu_isa"], row["precision"],
               row["suite"], row["kind"], row["comparison"],
               row["case_id"])
        by_case[key].append(row)
    summary = []
    for key, vals in by_case.items():
        ref = vals[0]
        record = dict(ref)
        record["replicates"] = len(vals)
        record["regression_percent"] = median(
            [r["regression_percent"] for r in vals])
        record["speedup"] = median([r["speedup"] for r in vals])
        record["candidate_ns"] = median(
            [r["candidate_ns"] for r in vals])
        record["reference_ns"] = median(
            [r["reference_ns"] for r in vals])
        record["shards"] = sorted({v["shard"] for v in vals})
        summary.append(record)
    return summary


def fmt_case(row):
    return ("x".join(str(n) for n in row["shape"]) +
            f", axes={row['axes']}, {row['layout']}, "
            f"threads={row['threads']}, "
            f"{'forward' if row['forward'] else 'inverse'}, "
            f"{'inplace' if row['inplace'] else 'out-of-place'}" +
            (f", type={row['transform_type']}" if row["kind"]
             in ("dct", "dst") else ""))


def generate_report(summary, coverage, issues, expected_shards):
    lines = [
        "# DUCC FFT cross-ISA benchmark",
        "",
        "Timing ratios are paired **within the same runner**; no samples from "
        "different CPUs are divided. Cases are aggregated by median "
        "regression across available shard replicates.",
        "",
        "**Regression % = 100 × (candidate time / reference time − 1)**. "
        "A positive value is slower; a negative value is faster. "
        "Speedup = reference time / candidate time.",
        "",
        f"Observed matrix shards: **{len(coverage)}/{expected_shards}**.",
        "",
    ]
    required = {"candidate-v3", "upstream-v1", "upstream-v3"}
    for shard in sorted(coverage):
        missing = sorted(required - coverage[shard])
        if missing:
            issues.append(f"shard {shard} lacks required profiles {missing}")
    missing_shards = sorted(set(range(expected_shards)) - set(coverage))
    if missing_shards:
        issues.append(f"missing shards: {missing_shards}")
    if issues:
        lines += ["## Validation issues", ""]
        lines += [f"- {issue}" for issue in issues[:100]]
        lines.append("")
    cpu_groups = sorted({(r["cpu_model"], r["cpu_isa"]) for r in summary})
    lines += ["## Hardware coverage", "",
              "| CPU model | ISA | Matched cases | Matrix shards |",
              "|---|---|---:|---:|"]
    for model, isa in cpu_groups:
        matches = [r for r in summary if (r["cpu_model"], r["cpu_isa"])
                   == (model, isa)]
        shards = {s for r in matches for s in r["shards"]}
        lines.append(f"| {model} | {isa} | {len(matches)} | {len(shards)} |")
    lines += ["", "A v4 result is recorded only if the runner actually "
              "supports AVX-512 and confirms active PSABI level 4. "
              "AVX2-only runners are included in all v1/v3 comparisons.", ""]
    for model, isa in cpu_groups:
        lines += [f"## {model} — {isa}", ""]
        for precision in ("single", "double"):
            rows_p = [r for r in summary if r["cpu_model"] == model and
                      r["cpu_isa"] == isa and r["precision"] == precision]
            if not rows_p:
                continue
            lines += [f"### {precision} precision", ""]
            for suite in SUITES:
                kinds = sorted({r["kind"] for r in rows_p if r["suite"] == suite})
                for kind in kinds:
                    local = [r for r in rows_p if
                             r["suite"] == suite and r["kind"] == kind]
                    if not local:
                        continue
                    lines += [f"#### {suite} / {kind}", "",
                              "| Comparison | Cases | Median speedup | "
                              "P95 regression | Worst regression | "
                              "Cases >20% slower |",
                              "|---|---:|---:|---:|---:|---:|"]
                    for _, _, label in COMPARISONS:
                        vals = [r for r in local if r["comparison"] == label]
                        if not vals:
                            continue
                        regress = [v["regression_percent"] for v in vals]
                        lines.append(
                            f"| {label} | {len(vals)} | "
                            f"{median([v['speedup'] for v in vals]):.3f}x | "
                            f"{percentile(regress, 0.95):+.1f}% | "
                            f"{max(regress):+.1f}% | "
                            f"{sum(x > 20 for x in regress)} |")
                    lines.append("")
                    # Worst-case examples use same-ISA controls first.
                    primary = [r for r in local if r["comparison"] in (
                        "v3 / upstream v3", "v4 / upstream v4")]
                    worst = sorted(primary, key=lambda r:
                                   r["regression_percent"], reverse=True)[:5]
                    if worst:
                        lines += ["Worst five cases (matched ISA):", "",
                                  "| Comparison | Dtype | Case | Regression | Replicates |",
                                  "|---|---|---|---:|---:|"]
                        for row in worst:
                            lines.append(
                                f"| {row['comparison']} | {row['dtype']} | "
                                f"{fmt_case(row)} | "
                                f"{row['regression_percent']:+.1f}% | "
                                f"{row['replicates']} |")
                        lines.append("")
    if not cpu_groups:
        lines += ["No paired benchmark measurements available.", ""]
    lines += ["## Interpretation", "",
              "The 10–20% regression objective is a policy target, not a "
              "claim of compliance. Prioritize repeated same-ISA regressions "
              "above 20%, then inspect the worst cases per FFT family. "
              "Differences across CPU models cannot be used as paired "
              "speedups.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, coverage, issues = load_rows(args.root)
    pairs = paired_rows(rows)
    summary = consolidate(pairs)
    report = generate_report(summary, coverage, issues, args.expected_shards)
    (args.output / "summary.md").write_text(report)
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n")
    if pairs:
        with (args.output / "paired-results.csv").open("w", newline="") as f:
            columns = list(pairs[0].keys())
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(pairs)
    (args.output / "coverage.json").write_text(json.dumps(
        {"shards": {str(k): sorted(v) for k, v in coverage.items()},
         "expected_shards": args.expected_shards,
         "issues": issues, "paired_rows": len(pairs),
         "consolidated_rows": len(summary)}, indent=2) + "\n")
    print(report[:25000])
    if issues:
        raise SystemExit(f"{len(issues)} validation issue(s)")
    if not pairs:
        raise SystemExit("no paired benchmark results")


if __name__ == "__main__":
    main()
