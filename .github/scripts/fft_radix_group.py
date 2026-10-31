"""Aggregate only fully paired and source-verified FFT radix samples by CPU/ISA."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

def readcsv(f):
    with f.open(newline="") as fp: return list(csv.DictReader(fp))

def writecsv(path,records,fields):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(records)

def gm(v):
    v=[x for x in v if x>0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else float("nan")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    runs=defaultdict(dict)
    inventory=[]
    for f in sorted(Path(args.input).rglob("summary.csv")):
        try:
            m=json.loads((f.parent/"metadata.json").read_text())
            s=json.loads((f.parent/"source_manifest.json").read_text())
            rows=readcsv(f)
            if int(m["cases"])!=len(rows) or len({v["case_id"] for v in rows})!=len(rows):
                raise ValueError("incomplete or duplicate case coverage")
            key=(f.parent.parent.parent.name,m["cpu"],int(s["isa"]))
            if s["label"] in runs[key]:
                raise ValueError("duplicate profile for runner")
            runs[key][s["label"]]={r["case_id"]:r for r in rows}
            inventory.append(dict(runner=key[0],cpu=key[1],isa=key[2],
                                  profile=s["label"],cases=len(rows),sha=s["sha"],status="accepted"))
        except Exception as e:
            inventory.append(dict(runner=f.parent.parent.parent.name,cpu="",
                                  isa="",profile=f.parent.name,cases=0,sha="",
                                  status="REJECTED: "+str(e)))
    per_runner=[]
    per_case=defaultdict(list)
    for (runner,cpu,isa),values in runs.items():
        for candidate,reference in (("R8","BASE"),("R48","BASE"),("R48","R8")):
            if candidate not in values or reference not in values:
                continue
            a,b=values[candidate],values[reference]
            if a.keys()!=b.keys():
                inventory.append(dict(runner=runner,cpu=cpu,isa=isa,profile=candidate,
                                      cases=0,sha="",status="REJECTED: mismatched cases"))
                continue
            groups=defaultdict(list)
            for case in a:
                x,y=float(a[case]["median_ns"]),float(b[case]["median_ns"])
                if x<=0 or y<=0: continue
                ratio=y/x
                row=a[case]
                for group in [row["family"],row["family"]+":"+row["dtype"],
                              row["family"]+":"+row["dtype"]+":"+("b1" if int(row["batch"])==1 else
                                  "b2_7" if int(row["batch"])<=7 else
                                  "b8_15" if int(row["batch"])<=15 else
                                  "b16_31" if int(row["batch"])<=31 else "b32_64")]:
                    groups[group].append(ratio)
                per_case[(cpu,isa,candidate+"/"+reference,row["family"],row["dtype"],case)].append(ratio)
            for family,ratios in groups.items():
                per_runner.append(dict(runner=runner,cpu=cpu,isa=isa,
                                       comparison=candidate+"/"+reference,
                                       family=family,cases=len(ratios),speedup_gm=gm(ratios)))
    group=defaultdict(list)
    for r in per_runner:
        group[(r["cpu"],r["isa"],r["comparison"],r["family"])].append(float(r["speedup_gm"]))
    summary=[]
    for (cpu,isa,comparison,family),vals in sorted(group.items()):
        summary.append(dict(cpu=cpu,isa=isa,comparison=comparison,family=family,
                            runners=len(vals),median_speedup=median(vals),
                            min_speedup=min(vals),max_speedup=max(vals)))
    regress=defaultdict(lambda: defaultdict(int))
    for (cpu,isa,comparison,family,dtype,case),ratios in per_case.items():
        r=regress[(cpu,isa,comparison,family,dtype)]
        r["cases"]+=1
        val=median(ratios)
        for th in (5,10,20):
            if val<1/(1+th/100):r[f"regression_gt_{th}"]+=1
    regrows=[dict(cpu=cpu,isa=isa,comparison=comp,family=family,dtype=dtype,**vals)
             for (cpu,isa,comp,family,dtype),vals in sorted(regress.items())]
    writecsv(output/"artifact_inventory.csv",inventory,
             ["runner","cpu","isa","profile","cases","sha","status"])
    writecsv(output/"per_runner.csv",per_runner,
             ["runner","cpu","isa","comparison","family","cases","speedup_gm"])
    writecsv(output/"cpu_summary.csv",summary,
             ["cpu","isa","comparison","family","runners","median_speedup",
              "min_speedup","max_speedup"])
    writecsv(output/"cpu_regressions.csv",regrows,
             ["cpu","isa","comparison","family","dtype","cases",
              "regression_gt_5","regression_gt_10","regression_gt_20"])
    header=["# Radix-8 vs radix-4+8 FFT experiment", "",
            f"Accepted profile artifacts: {sum('accepted'==v['status'] for v in inventory)}",
            f"Rejected profile artifacts: {sum(v['status']!='accepted' for v in inventory)}",
            "Ratios >1 mean the experimental profile is faster.",
            "Grouping uses the median of within-run geometric mean speedups;",
            "CPU models and ISA levels are never pooled.", "",
            "| CPU | ISA | Comparison | Family | Runners | Median | Range |",
            "|---|---|---|---|---:|---:|---|"]
    for r in summary:
        header.append(f"| {r['cpu']} | v{r['isa']} | {r['comparison']} | "
                      f"{r['family']} | {r['runners']} | "
                      f"{r['median_speedup']:.3f}x | {r['min_speedup']:.3f}–"
                      f"{r['max_speedup']:.3f}x |")
    header.extend(["","Interpret AVX2/v2 results as build and dispatch controls:",
                   "the new intrinsics are only compiled into v4 variants.",
                   "Check raw per-case repeats before classifying close differences."])
    (output/"REPORT.md").write_text("\n".join(header)+"\n")
    print("\n".join(header[:12]))
if __name__=="__main__":main()
