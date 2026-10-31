"""Run the 3 pinned DUCC source profiles with v2/v3/v4 runtime multiarch."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BENCH=ROOT/"python/demos/fft_bench.py"

def invoke(args,cwd=None,env=None,log=None,timeout=None):
    args=[str(s) for s in args]
    print("+"," ".join(args),flush=True)
    with (open(log,"w") if log else open(os.devnull,"w")) as f:
        x=subprocess.run(args,cwd=cwd,env=env,stdout=f,
                         stderr=subprocess.STDOUT,timeout=timeout)
    if x.returncode:
        raise RuntimeError(f"command exit {x.returncode}; log={log}")
    return x

def cpu():
    txt=Path("/proc/cpuinfo").read_text().splitlines()
    model=next((s.split(":",1)[1].strip() for s in txt
                if s.startswith("model name")),"unknown")
    flags=set(next((s.split(":",1)[1].split() for s in txt
                    if s.startswith("flags")),[]))
    levels=[2]
    if {"avx2","fma","bmi1","bmi2","f16c"}.issubset(flags):
        levels.append(3)
        if {"avx512f","avx512cd","avx512bw","avx512dq","avx512vl"}.issubset(flags):
            levels.append(4)
    return dict(model=model,levels=levels,affinity=sorted(os.sched_getaffinity(0)),
                flags=sorted(flags))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--samples",type=int,default=9)
    p.add_argument("--sample-ms",type=float,default=0.35)
    p.add_argument("--slot",type=int,required=True)
    args=p.parse_args(argv)
    root=Path(args.output).resolve()
    logs=root/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/"fft_radix_lock.json").read_text())
    info=cpu()
    status=dict(slot=args.slot,cpu=info,profiles=[],started=time.time(),
                source_lock=lock)
    def save():
        (root/"status.json").write_text(json.dumps(status,indent=2))
    save()
    env0=dict(os.environ)
    env0.update(DUCC0_OPTIMIZATION="multiarch-strip",OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                PIP_DISABLE_PIP_VERSION_CHECK="1")
    work=root/"work"
    work.mkdir(exist_ok=True)
    # Rotate profile order across independent runners to reduce time-order bias.
    profiles=lock["profiles"]
    profiles=profiles[(args.slot-1)%3:]+profiles[:(args.slot-1)%3]
    for prof in profiles:
        label=prof["label"]
        entry=dict(label=label,sha=prof["sha"],branch=prof["branch"],
                   status="pending",isas=[])
        status["profiles"].append(entry)
        save()
        try:
            source=work/("source-"+label)
            invoke(["git","worktree","add","--detach",source,prof["sha"]],
                   cwd=ROOT,log=logs/f"{label}-checkout.log",timeout=180)
            actual=subprocess.check_output(["git","rev-parse","HEAD"],
                                            cwd=source,text=True).strip()
            if actual!=prof["sha"]:
                raise RuntimeError("source sha mismatch")
            wheels=work/("wheels-"+label)
            wheels.mkdir(exist_ok=True)
            invoke([sys.executable,"-m","pip","wheel","--no-build-isolation",
                    "--no-deps","--no-cache-dir","-w",wheels,"."],
                   cwd=source,env=env0,log=logs/f"{label}-build.log",timeout=1800)
            candidates=list(wheels.glob("ducc0-*.whl"))
            if len(candidates)!=1:
                raise RuntimeError("one ducc wheel expected")
            install=work/("install-"+label)
            invoke([sys.executable,"-m","pip","install","--no-deps",
                    "--target",install,str(candidates[0])],
                   cwd=source,env=env0,log=logs/f"{label}-install.log",timeout=600)
            for isa in info["levels"]:
                name=f"{label}-v{isa}"
                phase=dict(isa=isa,status="pending")
                entry["isas"].append(phase)
                save()
                env=env0.copy()
                env["PYTHONPATH"]=str(install)
                env["DUCC0_MAX_PSABI_LEVEL"]=str(isa)
                try:
                    verify=("import ducc0; v=ducc0.misc.psabi_level(); "
                            f"assert v=={isa},(v,{isa}); print('selected',v)")
                    invoke([sys.executable,"-c",verify],cwd=source,env=env,
                           log=logs/f"{name}-isa.log",timeout=120)
                    if isa==max(info["levels"]):
                        invoke([sys.executable,"-m","pytest",
                                "python/test/test_fft.py","-q","-x"],
                               cwd=source,env=env,log=logs/f"{name}-tests.log",
                               timeout=1200)
                    dest=root/"benchmarks"/name
                    dest.mkdir(parents=True,exist_ok=True)
                    cmd=["taskset","-c",str(info["affinity"][0]),sys.executable,
                         str(BENCH),"--radix-suite","--output",dest,
                         "--profile",name,"--samples",str(args.samples),
                         "--sample-ms",str(args.sample_ms)]
                    invoke(cmd,cwd=ROOT,env=env,
                           log=logs/f"{name}-bench.log",timeout=10800)
                    (dest/"source_manifest.json").write_text(json.dumps({
                        "label":label,"branch":prof["branch"],
                        "sha":prof["sha"],"isa":isa,
                        "slot":args.slot},indent=2))
                    raw=dest/"raw.csv"
                    with raw.open("rb") as inp, gzip.open(str(raw)+".gz","wb",compresslevel=5) as out:
                        shutil.copyfileobj(inp,out)
                    raw.unlink()
                    phase["status"]="success"
                except Exception as e:
                    phase["status"]="failed"
                    phase["error"]=str(e)
                    print("FAILED",name,e,flush=True)
                finally:
                    save()
            entry["status"]="success" if any(v["status"]=="success" for v in entry["isas"]) else "failed"
        except Exception as e:
            entry["status"]="failed"
            entry["error"]=str(e)
            print("BUILD FAILED",label,e,flush=True)
        finally:
            entry["finished"]=time.time()
            save()
    status["finished"]=time.time()
    save()
    ok=sum(y["status"]=="success" for y in status["profiles"])
    print("successfully benchmarked profiles:",ok,"/3")
    return 0 if ok==3 else 1

if __name__=="__main__":
    raise SystemExit(main())
