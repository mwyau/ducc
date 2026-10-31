"""Reproducible radix-8 versus radix-4+8 FFT benchmark.

All power-of-four/eight products through 32768 are included, with every
1D batch 1..64 for both complex precisions and C/strided layouts. ND
cases exercise shared FFT stages. The nine raw samples remain inspectable.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import ducc0

BATCH_BOUNDARY = (1,2,3,4,7,8,9,15,16,17,31,32,33,63,64)
ND2 = ((4,8),(8,8),(16,16),(32,32),(64,64),(128,128),
       (256,256),(128,256),(32,256),(16,1024),(4,8192))
ND3 = ((8,8,8),(16,16,16),(4,16,64),(8,16,128),(16,32,64))
MAX_BYTES = 64 * 1024 * 1024

def lengths():
    vals={2}
    for a in range(10):
        for b in range(7):
            n=4**a*8**b
            if 4<=n<=32768:
                vals.add(n)
    return tuple(sorted(vals))

def cases():
    for n in lengths():
        for batch in range(1,65):
            for dtype in ("complex64","complex128"):
                for layout in ("C","strided"):
                    # Both standard and out-buffer cases at the critical width boundaries
                    for output in (("new","fixed") if batch in BATCH_BOUNDARY else ("new",)):
                        yield ("1d",(n,),batch,dtype,layout,output,True)
    for family,shapes in (("2d",ND2),("3d",ND3)):
        for shape in shapes:
            for batch in BATCH_BOUNDARY:
                if 16*batch*math.prod(shape)>MAX_BYTES:
                    continue
                for dtype in ("complex64","complex128"):
                    for layout in ("C","F"):
                        for output in ("new","fixed"):
                            yield (family,shape,batch,dtype,layout,output,True)

def make_array(family,shape,batch,dtype,layout,output):
    if layout=="strided":
        assert family=="1d"
        fullshape=(shape[0],batch)
        axes=(0,)
        a=np.empty(fullshape,dtype=dtype,order="C")
    else:
        fullshape=(batch,)+tuple(shape)
        axes=tuple(range(1,len(fullshape)))
        a=np.empty(fullshape,dtype=dtype,order=layout)
    a.fill(0.375 + 0.125j)
    a.flat[0]=complex(-0.6,0.8)
    out=np.empty_like(a,order="K") if output=="fixed" else None
    return a,out,axes

def measure(case,repeats,target_ms,do_correctness):
    family,shape,batch,dtype,layout,output,forward=case
    a,out,axes=make_array(family,shape,batch,dtype,layout,output)
    kwargs=dict(axes=axes,forward=forward,nthreads=1,out=out)
    def execute():
        return ducc0.fft.c2c(a,**kwargs)
    if do_correctness:
        got=execute()
        ref=np.fft.fftn(a,axes=axes)
        tol=3e-5 if dtype=="complex64" else 4e-12
        np.testing.assert_allclose(got,ref,rtol=tol,atol=tol*math.prod(shape))
    for _ in range(3): execute()
    st=time.perf_counter_ns()
    execute()
    single=max(time.perf_counter_ns()-st,1)
    loops=max(1,min(64,round(target_ms*1e6/single)))
    results=[]
    for _ in range(repeats):
        start=time.perf_counter_ns()
        for unused in range(loops):
            execute()
        results.append((time.perf_counter_ns()-start)/loops)
    return results,loops

def metadata():
    data=Path("/proc/cpuinfo").read_text().splitlines()
    cpu=next((t.split(":",1)[1].strip() for t in data if t.startswith("model name")),"unknown")
    caches=[]
    for p in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        try:
            caches.append({f:(p/f).read_text().strip()
                           for f in ("level","type","size","shared_cpu_list")})
        except OSError:
            pass
    return dict(cpu=cpu,caches=caches,affinity=sorted(os.sched_getaffinity(0)),
                psabi=int(ducc0.misc.psabi_level()),
                numpy=np.__version__,python=sys.version,host=platform.platform())

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--profile",required=True)
    p.add_argument("--samples",type=int,default=9)
    p.add_argument("--sample-ms",type=float,default=0.35)
    p.add_argument("--check-every",type=int,default=113)
    args=p.parse_args(argv)
    if args.samples<3: p.error("samples must be >=3")
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    info=metadata()
    info.update(profile=args.profile,samples=args.samples,sample_ms=args.sample_ms,
                lengths=lengths(),batch_range=[1,64],power4_power8=True)
    fields=["profile","case_id","family","shape","batch","dtype","layout","output",
            "repeat","ns","loops"]
    sf=[c for c in fields if c not in ("repeat","ns","loops")]
    sf+=["median_ns","min_ns","max_ns","mad_ns","samples","loops"]
    with (folder/"raw.csv").open("w",newline="") as f, \
         (folder/"summary.csv").open("w",newline="") as g:
        raw=csv.DictWriter(f,fieldnames=fields)
        summ=csv.DictWriter(g,fieldnames=sf)
        raw.writeheader();summ.writeheader()
        n=0;families={}
        for idx,case in enumerate(cases()):
            family,shape,batch,dtype,layout,output,forward=case
            case_id=f"{family}:{'x'.join(map(str,shape))}:b{batch}:{dtype}:{layout}:{output}"
            required_check=(idx%args.check_every==0 or
                            (batch in (1,8,16,64) and
                             shape in ((1024,),(4096,),(32768,),(128,256),(16,16,16))))
            times,loops=measure(case,args.samples,args.sample_ms,required_check)
            meta=dict(profile=args.profile,case_id=case_id,family=family,
                      shape="x".join(map(str,shape)),batch=batch,dtype=dtype,
                      layout=layout,output=output)
            mid=statistics.median(times)
            for ri,t in enumerate(times):
                raw.writerow({**meta,"repeat":ri,"ns":t,"loops":loops})
            summ.writerow({**meta,"median_ns":mid,"min_ns":min(times),
                          "max_ns":max(times),
                          "mad_ns":statistics.median(abs(t-mid) for t in times),
                          "samples":len(times),"loops":loops})
            n+=1
            families[family]=families.get(family,0)+1
            if n%250==0:
                f.flush();g.flush()
                print(args.profile,n,"completed",flush=True)
    info["cases"]=n
    info["families"]=families
    (folder/"metadata.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(dict(profile=args.profile,cases=n,families=families)),flush=True)

if __name__=="__main__":
    main()
