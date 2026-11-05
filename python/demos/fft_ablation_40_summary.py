#!/usr/bin/env python3
"""Aggregate independent FFT ablation groups: same-runner speedups by CPU."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

GROUPS = ("lanes", "scalar", "nocache", "long", "full")
SUITES = ("1d", "batched", "strided", "nd", "long",
          "real", "r2r", "convolution")
BASELINES = (1, 3, 4)


def med(seq):
    return statistics.median(seq)


def quantile(seq, q):
    x = sorted(seq)
    if not x:
        return float("nan")
    p = q * (len(x) - 1)
    lo = int(p)
    hi = min(lo + 1, len(x) - 1)
    return x[lo] * (1 - (p - lo)) + x[hi] * (p - lo)


def faster_mode(samples):
    x = sorted(float(t) for t in samples)
    if len(x) < 9:
        return med(x), False
    gaps = [(x[i] / max(x[i-1], 1e-9), i)
            for i in range(3, len(x)-1)]
    ratio, cut = max(gaps)
    if ratio >= 1.20 and len(x)-cut >= 2:
        return med(x[:cut]), True
    return med(x), False


def load(root):
    records, problems, shards = {}, [], set()
    fast_samples = 0
    for path in root.rglob("rows.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            shard = int(row["shard"])
            key = (shard, row["case_id"], row["profile"])
            if key in records:
                problems.append(f"duplicate row {key}")
                continue
            shards.add(shard)
            if row["status"] != "ok":
                problems.append(f"failed case {key}: {row.get('error','')[:140]}")
                continue
            fast, used = faster_mode(row["ns_samples"])
            fast_samples += int(used)
            row["selected_ns"] = fast
            records[key] = row
    for path in root.rglob("errors.json"):
        d = json.loads(path.read_text())
        if d.get("count"):
            problems.append(f"job failures in {path}: {d['failures'][:4]}")
    return records, problems, shards, fast_samples


def vm_clusters(records):
    by_cpu_case = defaultdict(list)
    for (shard, case_id, profile), row in records.items():
        if profile == "upstream-v3":
            by_cpu_case[(row["cpu_model"], case_id)].append(
                (shard, row["selected_ns"]))
    keep = {}
    details = []
    for key, points in by_cpu_case.items():
        if len(points) < 6:
            continue
        ordered = sorted(points, key=lambda p: p[1])
        possible = [(ordered[i][1]/max(ordered[i-1][1], 1e-9), i)
                    for i in range(2, len(ordered)-1)]
        if not possible:
            continue
        gap, cut = max(possible)
        if gap < 1.22 or len(ordered)-cut < 2:
            continue
        keep[key] = {shard for shard, _ in ordered[:cut]}
        details.append(dict(cpu=key[0], case=key[1], gap=gap,
                            retained=cut, dropped=len(ordered)-cut))
    return keep, details


def paired(records, keep):
    groups = defaultdict(list)
    for (shard, case_id, profile), row in records.items():
        group = profile.rsplit("-v", 1)[0]
        if group not in GROUPS:
            continue
        level = int(profile.rsplit("-v", 1)[1])
        allowed = keep.get((row["cpu_model"], case_id))
        if allowed is not None and shard not in allowed:
            continue
        for base_level in BASELINES:
            ref = records.get((shard, case_id, f"upstream-v{base_level}"))
            if ref is None:
                continue
            if row["cpu_model"] != ref["cpu_model"]:
                raise ValueError("mixed CPU within runner")
            key = (row["cpu_model"], row["cpu_isa"], case_id,
                   group, level, base_level)
            groups[key].append(dict(
                speedup=ref["selected_ns"]/row["selected_ns"],
                raw_speedup=ref["ns_median"]/row["ns_median"],
                shard=shard, row=row,
                ref_ns=ref["selected_ns"],
                candidate_ns=row["selected_ns"]))
    cases = []
    for (cpu, isa, cid, group, level, base), readings in groups.items():
        example = readings[0]["row"]
        cases.append(dict(
            cpu_model=cpu, cpu_isa=isa, case_id=cid, group=group,
            candidate_level=level, baseline_level=base,
            precision=example["precision"], dtype=example["dtype"],
            suite=example["suite"], kind=example["kind"],
            shape=example["shape"], layout=example["layout"],
            axes=example["axes"], threads=example["threads"],
            inplace=example["inplace"], forward=example["forward"],
            speedup=med([v["speedup"] for v in readings]),
            raw_speedup=med([v["raw_speedup"] for v in readings]),
            original_ns=med([v["ref_ns"] for v in readings]),
            candidate_ns=med([v["candidate_ns"] for v in readings]),
            runner_replicates=len(readings),
            shards=[v["shard"] for v in readings]))
    return cases


def grouped(cases):
    groups = defaultdict(list)
    for row in cases:
        key = (row["cpu_model"], row["cpu_isa"], row["precision"],
               row["suite"], row["kind"], row["group"],
               row["candidate_level"], row["baseline_level"])
        groups[key].append(row)
    out = []
    for (cpu, isa, precision, suite, kind, group, level, base), rows in groups.items():
        values = [r["speedup"] for r in rows]
        out.append(dict(cpu_model=cpu, cpu_isa=isa, precision=precision,
                        suite=suite, kind=kind, group=group,
                        candidate_level=level, baseline_level=base,
                        n_cases=len(rows), speedup=med(values),
                        p05_speedup=quantile(values, .05),
                        minimum_speedup=min(values),
                        slower_20pct=sum(x < (1/1.20) for x in values),
                        median_replicates=med(
                            [r["runner_replicates"] for r in rows])))
    return out


def save_csv(path, items, columns):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)


def label(x):
    return f"{x:.3f}x" if x is not None else "—"


def report(summary, cases, shards, problems, modes, clusters, expected):
    cpu_models = sorted({r["cpu_model"] for r in summary})
    lookup = {(r["cpu_model"], r["precision"], r["suite"], r["kind"],
               r["group"], r["candidate_level"], r["baseline_level"]): r
              for r in summary}
    lines = [
        "# FFT optimization ablation: speedups versus original v1/v3/v4",
        "",
        "**Speedup = original runtime / candidate runtime.** "
        "1.2x means 20% faster, while 0.8x means 25% slower.",
        "",
        f"Observed complete-suite runner IDs: {len(shards)}/{expected}. "
        f"Aggregated {len(cases):,} CPU/case/profile/baseline comparisons.",
        "",
        "Each runner builds original upstream, four isolated changes and "
        "all changes together; it executes all eight FFT suites, with 11 "
        "timing samples per case/profile. Timing order rotates per runner.",
        "",
        "Isolated groups: lanes = batch width and SIMD tails; "
        "scalar = AVX-512 generic radix and radix-8; "
        "nocache = stack-owned 1D plan with NO_FFT_CACHE macro enabled; "
        "long = intra-transform and long-batch policy including linkage fix; "
        "full = all improvements in ordinary/default caching configuration.",
        "",
        "## VM bimodality rule", "",
        f"Fast-mode splits within 11-sample sets: {modes}. "
        f"Cross-run bimodal original-v3 CPU/case clusters: {len(clusters)}.",
        "",
        "For a runner time series, select the faster cluster only for "
        ">=20% separation and at least three fast and two slow samples. "
        "Across independent runners, require >=6 observations on a CPU "
        "model, >=22% gap in ORIGINAL v3 timings and at least two "
        "runners in each mode. The same retained runner set is used "
        "for all comparisons of that CPU/case. Otherwise use all data. "
        "This filtering cannot independently prove VM scheduling as the cause.",
        "",
        "## CPU / precision / FFT family / optimization group", "",
        "Columns report median speedup versus original v1, v3, and v4. "
        "The last two columns use same-ISA original comparisons.", ""
    ]
    for cpu in cpu_models:
        cpu_cases = [r for r in cases if r["cpu_model"] == cpu]
        isa = cpu_cases[0]["cpu_isa"]
        n_runners = len({s for r in cpu_cases for s in r["shards"]})
        lines += [f"### {cpu} ({isa}, {n_runners} runners)", ""]
        for precision in ("single", "double"):
            if not any(r["cpu_model"] == cpu and r["precision"] == precision
                       for r in summary):
                continue
            lines += [f"#### {precision} precision", "",
                      "| Family | Isolated change | ISA | Cases | "
                      "vs original v1 | vs original v3 | vs original v4 | "
                      "Worst same ISA | >20% slower |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for suite in SUITES:
                kinds = sorted({r["kind"] for r in summary
                                if r["cpu_model"] == cpu and
                                r["precision"] == precision and
                                r["suite"] == suite})
                for kind in kinds:
                    for group in GROUPS:
                        for level in (3, 4):
                            vals = [lookup.get((cpu, precision, suite, kind,
                                               group, level, baseline))
                                    for baseline in BASELINES]
                            record = next((x for x in vals if x), None)
                            if record is None:
                                continue
                            same = vals[1] if level == 3 else vals[2]
                            cell = [label(r["speedup"]) if r else "—"
                                    for r in vals]
                            lines.append(
                                f"| {suite}/{kind} | {group} | v{level} | "
                                f"{record['n_cases']} | " +
                                " | ".join(cell) + " | " +
                                (label(same["minimum_speedup"])
                                 if same else "—") + " | " +
                                (f"{same['slower_20pct']}/"
                                 f"{same['n_cases']}" if same else "—") +
                                " |")
            lines.append("")
        worst = sorted([r for r in cpu_cases
                        if r["candidate_level"] == r["baseline_level"]],
                       key=lambda r: r["speedup"])[:15]
        lines += ["Worst 15 individual same-ISA cases on this CPU:", "",
                  "| Group | ISA | Kind | Dtype | Shape | Layout | "
                  "Threads | Speedup | Runner repeats |",
                  "|---|---|---|---|---|---|---:|---:|---:|"]
        for r in worst:
            lines.append(
                f"| {r['group']} | v{r['candidate_level']} | "
                f"{r['suite']}/{r['kind']} | {r['dtype']} | "
                f"{'x'.join(map(str,r['shape']))} | {r['layout']} | "
                f"{r['threads']} | {r['speedup']:.3f}x | "
                f"{r['runner_replicates']} |")
        lines += [""]
    lines += ["## Validation", "",
              f"Validation issues: {len(problems)}.", ""]
    lines.extend(f"- {p}" for p in problems[:15])
    if len(problems) > 15:
        lines.append(f"- {len(problems)-15} more in coverage.json.")
    lines += ["", "Interpretation: original v1 comparisons include SIMD "
              "advantages present before the new branch; comparing equal "
              "ISA levels isolates the changes. Opt-in no-cache and "
              "default-cache full builds are intentionally different "
              "configurations. Small case counts do not establish robust "
              "worst-case guarantees.", ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-shards", type=int, default=40)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, problems, shards, modes = load(args.root)
    retained, clusters = vm_clusters(rows)
    cases = paired(rows, retained)
    summary = grouped(cases)
    missing = sorted(set(range(args.expected_shards)) - shards)
    if missing:
        problems.append(f"Missing runner IDs: {missing}")
    if not cases:
        problems.append("No paired cases found")
    (args.output / "report.md").write_text(
        report(summary, cases, shards, problems, modes, clusters,
               args.expected_shards))
    (args.output / "coverage.json").write_text(json.dumps(
        dict(observed_shards=sorted(shards), expected=args.expected_shards,
             case_rows=len(rows), aggregated_cases=len(cases),
             fast_modes=modes, runner_bimodal_clusters=clusters,
             problems=problems), indent=2) + "\n")
    save_csv(args.output / "cpu-family-speedups.csv", summary,
             ["cpu_model", "cpu_isa", "precision", "suite", "kind",
              "group", "candidate_level", "baseline_level", "n_cases",
              "speedup", "p05_speedup", "minimum_speedup",
              "slower_20pct", "median_replicates"])
    save_csv(args.output / "case-speedups.csv", cases,
             ["cpu_model", "cpu_isa", "case_id", "precision", "suite",
              "kind", "dtype", "shape", "axes", "layout", "threads",
              "inplace", "forward", "group", "candidate_level",
              "baseline_level", "speedup", "raw_speedup", "original_ns",
              "candidate_ns", "runner_replicates"])
    print((args.output / "report.md").read_text()[:20000], flush=True)
    if problems:
        raise SystemExit(f"{len(problems)} validation issues")


if __name__ == "__main__":
    main()
