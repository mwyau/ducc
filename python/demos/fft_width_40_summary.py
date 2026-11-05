#!/usr/bin/env python3
"""Compare one-line strided SIMD width candidate vs unchanged and upstream.

Raw paired speedup is the primary estimator. Optional fast-mode selection
is computed from upstream-v3 only; both estimates are retained to reveal
selection sensitivity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from fft_ablation_40_summary import load, med, quantile, SUITES


def compare(records, retained):
    grouped = defaultdict(list)
    for (shard, cid, profile), new in records.items():
        if profile not in ("original-v3", "original-v4",
                           "fixed-v3", "fixed-v4"):
            continue
        level = int(profile[-1])
        references = ["upstream-v1", "upstream-v3", "upstream-v4"]
        if profile.startswith("fixed"):
            references.append(f"original-v{level}")
        for reference in references:
            old = records.get((shard, cid, reference))
            if old is None:
                continue
            if (old["cpu_model"], old["cpu_isa"]) != (
                    new["cpu_model"], new["cpu_isa"]):
                raise ValueError("cannot pair different CPU/ISA")
            allowed = retained.get((new["cpu_model"], cid))
            if allowed is not None and shard not in allowed:
                continue
            key = (new["cpu_model"], new["cpu_isa"], cid,
                   profile, reference)
            grouped[key].append(dict(
                raw=old["ns_median"]/new["ns_median"],
                filtered=old["selected_ns"]/new["selected_ns"],
                shard=shard, old_ns=old["ns_median"],
                new_ns=new["ns_median"], example=new))
    cases = []
    for (cpu, isa, cid, profile, reference), pairs in grouped.items():
        example = pairs[0]["example"]
        cases.append(dict(
            cpu_model=cpu, cpu_isa=isa, case_id=cid,
            candidate=profile, reference=reference,
            suite=example["suite"], kind=example["kind"],
            dtype=example["dtype"], precision=example["precision"],
            shape=example["shape"], axes=example["axes"],
            layout=example["layout"], threads=example["threads"],
            forward=example["forward"], inplace=example["inplace"],
            speedup=med([r["raw"] for r in pairs]),
            filtered_speedup=med([r["filtered"] for r in pairs]),
            reference_ns=med([r["old_ns"] for r in pairs]),
            candidate_ns=med([r["new_ns"] for r in pairs]),
            replicates=len(pairs),
            shards=sorted({r["shard"] for r in pairs})))
    by_family = defaultdict(list)
    for row in cases:
        key = (row["cpu_model"], row["cpu_isa"], row["precision"],
               row["suite"], row["kind"], row["candidate"],
               row["reference"])
        by_family[key].append(row)
    summaries = []
    for (cpu, isa, precision, suite, kind, cand, ref), rows in by_family.items():
        speeds = [r["speedup"] for r in rows]
        adjusted = [r["filtered_speedup"] for r in rows]
        summaries.append(dict(
            cpu_model=cpu, cpu_isa=isa, precision=precision,
            suite=suite, kind=kind, candidate=cand, reference=ref,
            n_cases=len(rows), median_speedup=med(speeds),
            median_filtered_speedup=med(adjusted),
            p05_speedup=quantile(speeds, .05),
            worst_speedup=min(speeds),
            slower_10pct=sum(x < 1/1.10 for x in speeds),
            slower_20pct=sum(x < 1/1.20 for x in speeds),
            median_replicates=med([r["replicates"] for r in rows])))
    return cases, summaries


def csv_write(path, entries, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def format_case(row):
    return (f"{row['dtype']} {'x'.join(map(str, row['shape']))}, "
            f"axes={row['axes']}, {row['layout']}, "
            f"threads={row['threads']}, "
            f"{'forward' if row['forward'] else 'inverse'}, "
            f"{'in-place' if row['inplace'] else 'out-of-place'}")


def markdown(cases, summaries, shards, errors, modes, clusters, expected):
    cpus = sorted({(r["cpu_model"],r["cpu_isa"]) for r in summaries})
    lines = [
        "# Full-width strided SIMD policy: scientific FFT A/B benchmarks", "",
        "**Speedup = reference runtime / candidate runtime.** "
        "Values above 1x mean the candidate is faster.",
        "",
        "Pinned sources: upstream 757145c3; unchanged SIMD f80b9878; "
        "strided full-width candidate 36684e01. All builds use "
        "normal caching and multiarch-strip. Every runner covers "
        "the same scientific cases with 11 timing samples per case "
        "and rotated profile order.",
        "",
        "Raw paired speedup is the primary estimate. Within-series "
        "fast-mode speedup is a sensitivity measure. All available "
        "runners and cases are included; no baseline-fast runner "
        "selection is applied. CPU model and ISA are distinct groups.",
        "",
        f"Observed runner shards: **{len(shards)}/{expected}**; "
        f"consolidated comparisons: **{len(cases):,}**. "
        f"Within-series fast-mode splits: {modes}; "
        f"across-runner clusters excluded from analysis: {len(clusters)}.",
        "",
        "## Hardware coverage", "",
        "| CPU | ISA | Runners represented |",
        "|---|---|---:|",
    ]
    for cpu, isa in cpus:
        selected = [r for r in cases if r["cpu_model"] == cpu and r["cpu_isa"] == isa]
        ids = {s for r in selected for s in r["shards"]}
        lines.append(f"| {cpu} | {isa} | {len(ids)} |")
    lines += ["", "## Direct comparison: fix versus unchanged improvements", "",
              "This is the decision-making comparison: same CPU, same ISA, "
              "same FFT case, independent source builds.", ""]
    for cpu, isa in cpus:
        sub = [x for x in summaries if x["cpu_model"] == cpu and x["cpu_isa"] == isa and
               x["candidate"].startswith("fixed") and
               x["reference"].startswith("original")]
        if not sub:
            continue
        lines += [f"### {cpu} ({sub[0]['cpu_isa']})", "",
                  "| FFT family | Precision | ISA | Cases | "
                  "Raw median | Adjusted median | P05 raw | Worst raw | "
                  ">10% slower | >20% slower |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for suite in SUITES:
            for precision in ("single", "double"):
                section = sorted(
                    [x for x in sub if x["suite"] == suite and
                     x["precision"] == precision],
                    key=lambda x: (x["kind"], x["candidate"]))
                for x in section:
                    lines.append(
                        f"| {suite}/{x['kind']} | {precision} | "
                        f"{x['candidate'][-2:]} | {x['n_cases']} | "
                        f"{x['median_speedup']:.3f}x | "
                        f"{x['median_filtered_speedup']:.3f}x | "
                        f"{x['p05_speedup']:.3f}x | "
                        f"{x['worst_speedup']:.3f}x | "
                        f"{x['slower_10pct']} | {x['slower_20pct']} |")
        local = sorted(
            [x for x in cases if x["cpu_model"] == cpu and x["cpu_isa"] == isa and
             x["candidate"].startswith("fixed") and
             x["reference"].startswith("original")],
            key=lambda r: r["speedup"])
        lines += ["", "Worst 10 direct fix-versus-original cases:", "",
                  "| ISA | Case | Speedup | Filtered | Runner repeats |",
                  "|---|---|---:|---:|---:|"]
        for x in local[:10]:
            lines.append(
                f"| {x['candidate'][-2:]} | {x['suite']}/{x['kind']}: "
                f"{format_case(x)} | {x['speedup']:.3f}x | "
                f"{x['filtered_speedup']:.3f}x | {x['replicates']} |")
        lines.append("")
    lines += ["## Both candidate branches versus upstream v1, v3, v4",
              "", "Original and fixed are compared independently against "
              "each available upstream ISA level on the same runner.", ""]
    for cpu, isa in cpus:
        for precision in ("single", "double"):
            sub = [x for x in summaries if x["cpu_model"] == cpu and x["cpu_isa"] == isa and
                   x["precision"] == precision and
                   x["reference"].startswith("upstream")]
            if not sub:
                continue
            lines += [f"### {cpu} ({isa}) — {precision}", "",
                      "| FFT | Candidate | Ref | Cases | Median speedup | "
                      "P05 speedup | >20% slower |",
                      "|---|---|---|---:|---:|---:|---:|"]
            for x in sorted(sub, key=lambda z: (
                    SUITES.index(z["suite"]), z["kind"],
                    z["candidate"], z["reference"])):
                lines.append(
                    f"| {x['suite']}/{x['kind']} | {x['candidate']} | "
                    f"{x['reference']} | {x['n_cases']} | "
                    f"{x['median_speedup']:.3f}x | "
                    f"{x['p05_speedup']:.3f}x | "
                    f"{x['slower_20pct']} |")
            lines.append("")
    lines += ["## Targeted regression cases", "",
              "These cases are shown without silently selecting the "
              "best CPU/runner observations. Both fix and unchanged "
              "branch results are compared with the upstream same-ISA "
              "reference and with each other.", "",
              "| CPU | Comparison | FFT case | Raw speedup | "
              "Adjusted | Runner repeats |",
              "|---|---|---|---:|---:|---:|"]
    targeted = [x for x in cases if
                (x["dtype"] == "complex64" and
                 x["shape"] == [4, 32768] and x["axes"] == [0]) or
                (x["dtype"] == "complex128" and
                 x["shape"] == [131064])]
    for x in sorted(targeted, key=lambda x: (
            x["cpu_model"], x["candidate"], x["reference"])):
        if x["reference"].startswith("upstream") and (
                x["reference"][-1] != x["candidate"][-1]):
            continue
        lines.append(
            f"| {x['cpu_model']} | {x['candidate']} / "
            f"{x['reference']} | {format_case(x)} | "
            f"{x['speedup']:.3f}x | "
            f"{x['filtered_speedup']:.3f}x | "
            f"{x['replicates']} |")
    lines += ["", "## Validation", "",
              f"Reported problems: {len(errors)}.", ""]
    lines.extend(f"- {e}" for e in errors[:15])
    if len(errors) > 15:
        lines.append(f"- Additional {len(errors)-15} issues in coverage.json.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-shards", type=int, default=40)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, errors, shards, modes = load(args.root)
    retain, clusters = {}, []  # No selective runner exclusion.
    cases, groups = compare(records, retain)
    missing = sorted(set(range(args.expected_shards)) - shards)
    if missing:
        errors.append(f"Missing runner shards: {missing}")
    if not cases:
        errors.append("No paired benchmark cases")
    (args.output / "report.md").write_text(
        markdown(cases, groups, shards, errors,
                 modes, clusters, args.expected_shards))
    (args.output / "coverage.json").write_text(
        json.dumps(dict(shards=sorted(shards), expected=args.expected_shards,
                        rows=len(records), cases=len(cases),
                        groups=len(groups),
                        within_run_fast_modes=modes,
                        upstream_selected_clusters=clusters,
                        issues=errors), indent=2) + "\n")
    csv_write(args.output / "case-speedups.csv", cases,
              ["cpu_model", "cpu_isa", "case_id", "candidate", "reference",
               "suite", "kind", "precision", "dtype", "shape", "axes",
               "layout", "threads", "forward", "inplace", "speedup",
               "filtered_speedup", "reference_ns", "candidate_ns",
               "replicates"])
    csv_write(args.output / "cpu-category-speedups.csv", groups,
              ["cpu_model", "cpu_isa", "precision", "suite", "kind",
               "candidate", "reference", "n_cases", "median_speedup",
               "median_filtered_speedup", "p05_speedup", "worst_speedup",
               "slower_10pct", "slower_20pct", "median_replicates"])
    direct = [x for x in groups if
              x["candidate"].startswith("fixed") and
              x["reference"].startswith("original")]
    csv_write(args.output / "fix-vs-original.csv", direct,
              ["cpu_model", "cpu_isa", "precision", "suite", "kind",
               "candidate", "reference", "n_cases", "median_speedup",
               "median_filtered_speedup", "p05_speedup", "worst_speedup",
               "slower_10pct", "slower_20pct", "median_replicates"])
    print((args.output / "report.md").read_text()[:23000], flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} validation issues")


if __name__ == "__main__":
    main()
