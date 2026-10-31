"""Aggregate only complete, pinned FFT runner artifacts by exact CPU model and ISA."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def geom_mean(values):
    values = [v for v in values if math.isfinite(v) and v > 0]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else math.nan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--planned", type=int, default=16)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.input)
    lock_path = Path(__file__).resolve().parents[2] / "fft_policy_lock.json"
    pinned = {p["label"]: p for p in json.loads(lock_path.read_text())["profiles"]}
    permitted_families = {"1d": {"1d"}, "nd": {"2d", "3d"},
                          "long": {"long"}}
    refs = defaultdict(dict)
    inventory = []
    for fn in sorted(root.rglob("summary.csv")):
        folder = fn.parent
        try:
            meta = json.loads((folder / "metadata.json").read_text())
            src = json.loads((folder / "source_manifest.json").read_text())
            rows = load_csv(fn)
            suite = src.get("suite")
            if suite not in permitted_families or suite != meta.get("suite"):
                raise ValueError("Suite missing or inconsistent between manifest and metadata")
            label = src["label"]
            if label not in pinned or src["sha"] != pinned[label]["sha"]:
                raise ValueError("Source revision differs from pinned lock")
            if src.get("compiler_defines", {}) != pinned[label].get("compiler_defines", {}):
                raise ValueError("Compiler defines differ from pinned lock")
            if int(meta.get("samples", -1)) != 9:
                raise ValueError("Expected nine timing repeats per case")
            if not (folder / "raw_repeats.csv.gz").is_file():
                raise ValueError("Missing raw repeat archive")
            if any(r["family"] not in permitted_families[suite] for r in rows):
                raise ValueError("Case family differs from requested suite")
            if any(int(r["repeats"]) != 9 or float(r["median_ns"]) <= 0 for r in rows):
                raise ValueError("Missing timing repeats or nonpositive timing")
            if not rows or len(rows) != int(meta["cases_total"]):
                raise ValueError("Missing or incomplete per-case CSV")
            keys = {r["case_id"] for r in rows}
            if len(keys) != len(rows):
                raise ValueError("Duplicate case keys")
            runner = folder.parents[1].name  # suite-qualified artifact name
            if not runner.startswith(f"fft-policy-runner-{suite}-"):
                raise ValueError("Artifact runner name does not match suite")
            profile = src["label"]
            isa = int(src["isa"])
            if profile == "BASE":
                profile = "V" + str(isa)
            expected = profile + "-v" + str(isa) if src["label"] != "BASE" else "BASE-v" + str(isa)
            if folder.name != expected:
                raise ValueError("Profile/ISA metadata mismatch")
            mid = (runner, meta["cpu_model"], isa)
            if profile in refs[mid]:
                raise ValueError("Duplicate runner profile")
            refs[mid][profile] = {r["case_id"]: (float(r["median_ns"]), r["family"]) for r in rows}
            inventory.append(dict(runner=runner, cpu_model=meta["cpu_model"],
                                  isa=isa, profile=profile, sha=src["sha"],
                                  total_cases=len(rows), status="accepted", reason=""))
        except Exception as e:
            inventory.append(dict(runner=fn.parts[-4] if len(fn.parts) >= 4 else "unknown",
                                  cpu_model="", isa="", profile=folder.name, sha="",
                                  total_cases=0, status="rejected", reason=str(e)))
    by_run = []
    by_case = defaultdict(list)
    for (runner, cpu, isa), profiles in refs.items():
        baseline = profiles.get("V" + str(isa))
        if not baseline:
            continue
        for profile, cases in profiles.items():
            if profile.startswith("V"):
                continue
            shared = baseline.keys() & cases.keys()
            if shared != baseline.keys() or shared != cases.keys():
                inventory.append(dict(runner=runner, cpu_model=cpu, isa=isa,
                                      profile=profile, sha="", total_cases=len(shared),
                                      status="rejected", reason="Different case coverage from baseline"))
                continue
            groups = defaultdict(list)
            for key in shared:
                base_ns, family = baseline[key]
                test_ns, _ = cases[key]
                if base_ns > 0 and test_ns > 0:
                    ratio = base_ns / test_ns
                    groups[family].append(ratio)
                    if family == "1d":
                        shape = key.split(":")[1]
                        if shape in ("1024", "2048", "4096"):
                            groups["1d_priority"].append(ratio)
                    by_case[(cpu, isa, profile, family, key)].append(ratio)
            for family, ratios in groups.items():
                if ratios:
                    by_run.append(dict(runner=runner, cpu_model=cpu, isa=isa,
                                       profile=profile, family=family,
                                       ncases=len(ratios), gm_speedup=geom_mean(ratios)))
    by_cpu = defaultdict(list)
    for row in by_run:
        key = (row["cpu_model"], row["isa"], row["profile"], row["family"])
        by_cpu[key].append(row["gm_speedup"])
    cpu_rows = []
    for (cpu, isa, profile, family), values in sorted(by_cpu.items()):
        cpu_rows.append(dict(cpu_model=cpu, isa=isa, profile=profile, family=family,
                             runners=len(values), median_speedup=median(values),
                             min_speedup=min(values), max_speedup=max(values),
                             small_sample=int(len(values) < 3)))
    slowdowns = defaultdict(lambda: defaultdict(int))
    for (cpu, isa, profile, family, key), ratios in by_case.items():
        med = median(ratios)
        rec = slowdowns[(cpu, isa, profile, family)]
        rec["cases"] += 1
        for threshold in (5, 10, 20):
            if med < 1 / (1 + threshold / 100):
                rec["regress_gt_" + str(threshold)] += 1
    regress = []
    for (cpu, isa, profile, family), rec in sorted(slowdowns.items()):
        regress.append({"cpu_model": cpu, "isa": isa, "profile": profile, "family": family,
                        **rec})
    write_csv(out / "artifact_inventory.csv", inventory,
              ["runner", "cpu_model", "isa", "profile", "sha", "total_cases", "status", "reason"])
    write_csv(out / "per_runner_geomeans.csv", by_run,
              ["runner", "cpu_model", "isa", "profile", "family", "ncases", "gm_speedup"])
    write_csv(out / "cpu_model_medians.csv", cpu_rows,
              ["cpu_model", "isa", "profile", "family", "runners",
               "median_speedup", "min_speedup", "max_speedup", "small_sample"])
    write_csv(out / "cpu_regression_counts.csv", regress,
              ["cpu_model", "isa", "profile", "family", "cases",
               "regress_gt_5", "regress_gt_10", "regress_gt_20"])
    accepted = sum(r["status"] == "accepted" for r in inventory)
    rejected = len(inventory) - accepted
    lines = ["# DUCC FFT policy CPU-model analysis", "",
             f"Accepted runner/profile artifacts: {accepted}; rejected: {rejected}",
             f"Expected candidate runner jobs across three suites: {args.planned}", "",
             "All comparisons are paired to the same runner's untouched DUCC V3 or V4",
             "reference at the same ISA; no absolute times are pooled between CPU models.",
             "Ratios >1 are faster. Each cell is the median of within-run geometric means.",
             "", "| CPU model | ISA | Profile | Family | Runs | Median | Range |",
             "|---|---|---|---|---:|---:|---|"]
    for r in cpu_rows:
        lines.append(f"| {r['cpu_model']} | v{r['isa']} | {r['profile']} | "
                     f"{r['family']} | {r['runners']}{'*' if r['small_sample'] else ''} | "
                     f"{r['median_speedup']:.3f}x | "
                     f"{r['min_speedup']:.3f}–{r['max_speedup']:.3f}x |")
    lines.extend(["", "* Small-sample CPU groups (<3 independent runner jobs).",
                  "F8 and F16 bypass the scalar and width L2 checks; they are diagnostics,",
                  "not production candidates. Consult per-run CSV and regression counts.",
                  "E = G-1024k-S0-T1; G = G-1024k-S1-T1.",
                  "The long N8192 batch<=64 family is excluded from this sweep.",
                  ""])
    (out / "CPU_GROUPED_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
