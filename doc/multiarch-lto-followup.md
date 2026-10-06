# LTO and multiarch follow-up

Date: 2026-10-06. This follow-up records the additional compiler, ISA-boundary, code-generation, and repeat-benchmark investigations for the multiarch dispatch work. The PR worktree remains clean at `e41fa307d45b`; no changes were pushed or made to the PR branch.

## Recommendation

Keep the current A implementation on the PR branch for now. The latest explicit-environment focused benchmarks put B close to A, but B still depends on compiler- and link-order-sensitive LTO behavior to keep baseline code free of higher-ISA instructions. GCC 14 and Clang both show startup ISA leakage, and GCC 15 shows leakage when profile-object order changes. The FFT-only noLTO hybrid also has near-parity focused timings, but it has only been audited with GCC 15 in the default order; the remaining profile objects still use LTO.

The native control establishes that LTO can slow some FFT shapes even without multiarch, and GCC's missed-inline report and one-kernel disassembly identify a plausible mechanism. The later same-session focused A/B run put B near A. A subsequent ABBA/BAAB crossover showed B slower, but unrelated CPU load overlapped it, so that run is not a clean reproduction. The original large B FFT penalty is therefore not established as stable, and the cross-run discrepancy remains unresolved. Candidate C was not implemented, consistent with the requested scope.

## Host and method

- Host: AMD Ryzen 9 5950X, 16 cores / 32 hardware threads. The host supports x86-64-v1 and v3/AVX2 but not v4/AVX512. **No v4 code was executed**; v4 checks were static only.
- Benchmark affinity: CPUs 16–31; CPU governor and EPP were set to `performance`. OpenMP, OpenBLAS, MKL, and NumExpr pools were limited to one unless DUCC's explicit thread count was under test.
- Main toolchain: GCC 15.2.0, GNU ld/objdump 2.46, Python 3.13.13, NumPy 2.5.3, pybind11 3.1.0, pyperf 2.10.0, CMake 4.4.4, Ninja 1.13.2. Additional checks used GCC 14.3.0 and Clang/LLD 21.1.8.
- The latest A/B full matrix used `pyperf --copy-env`; the focused harness also checked that each worker selected the requested cap. Focused runs used five worker processes and seven measured values per process. The full matrix used three measured worker runs with three values each. Hybrid full-matrix results were collected earlier and are not interleaved with the later A/B rerun.
- Hardware performance counters could not be collected. After loading the extracted `perf` runtime libraries, `perf stat` still returned “No supported events found”; `/proc/sys/kernel/perf_event_paranoid` is `4`. No system setting was changed. `py-spy` sampling was used as a fallback.

## Builds, sizes, and correctness

A/B/D figures below are the earlier clean GCC 15 production build measurements; the hybrid was measured in a separate clean release build. RSS is sampled aggregate descendant RSS, except the single-process GNU `time` maximum noted in the raw log.

| Variant | Clean wheel build | CPU time | Peak aggregate RSS | Wheel | Extension | `.text` |
|---|---:|---:|---:|---:|---:|---:|
| A: current multiarch | 114.2 s median (5) | 1,844 s median | 24,490 MiB | 14,777,486 B | 43,880,496 B | 38,145,296 B |
| B: profile IPO | 91.2 s median (5) | 1,785 s median | 18,769 MiB | 13,447,155 B | 40,089,360 B | 34,752,272 B |
| D: profile-local LTO | 293.5 s median (3) | 1,252 s median | 17,538 MiB | 13,391,217 B | 44,661,584 B | 39,208,336 B |
| B with FFT instantiations noLTO | 91.82 s (one sampled build) | 1,739.83 s | 19,338 MiB sampled; 1,283,500 KiB GNU `time` max | 13,128,878 B | 39,954,232 B | 34,637,072 B |

A, B, and D each passed the C++ driver (12/12) and full Python suite (57,507 passed, 4,632 skipped). The hybrid passed the same C++ driver and Python suite. At forced v1, the GCC 15 A/B/D/hybrid numerical checks matched A bitwise for FFT, SHT, NUFFT, and wgridder; forced v3 checks passed configured tolerances. Clang A/B smoke checks at forced v1/v3 also matched A within the configured checks. These runs do not validate v4 execution.

The hybrid changes only `src/ducc0/fft/fft_inst1.cc` and `fft_inst2.cc` to compile without LTO in each profile. Compile commands confirmed those six profile objects receive `-fno-lto`; the remaining profile objects retain LTO. The experiment is on local branch `exp/lto-followup-b-fft-nolto` at `4fa0090c9ff6`, not on the PR branch.

D's GCC 15 profile-local LTO path required `-flto-partition=none`; GCC ICEd with other tested partition modes during `-r -flto -flinker-output=nolto-rel`. This produced the longest build despite lower aggregate CPU time and RSS.

## Runtime: latest paired rerun and earlier result

### Latest focused FFT comparison

Medians are milliseconds. Ratios above 1 mean the candidate was slower than A. Each focused result has 35 measured values.

| Cap | Threads | A | B | FFT-noLTO hybrid | B/A | Hybrid/A |
|---:|---:|---:|---:|---:|---:|---:|
| v1 | 1 | 18.960 | 19.001 | 18.962 | 1.002 | 1.000 |
| v1 | 16 | 18.455 | 18.735 | 18.746 | 1.015 | 1.016 |
| v3 | 1 | 13.180 | 13.246 | 13.324 | 1.005 | 1.011 |
| v3 | 16 | 13.173 | 13.255 | 13.207 | 1.006 | 1.003 |

In this explicit-environment focused rerun, B is 0.2–1.5% slower than A and the hybrid is 0.0–1.6% slower. These differences are close to the run-to-run spread (CVs around 1–3%).

### Latest full eight-case matrix

Medians are milliseconds. This matrix was collected separately from the focused runs; CVs are available in the JSON summary. The hybrid results were captured before this later A/B rerun, so its column is a useful reference rather than a time-interleaved control.

| Cap | Workload | Threads | A | B | Hybrid | B/A | Hybrid/A |
|---:|---|---:|---:|---:|---:|---:|---:|
| v1 | FFT | 1 | 19.481 | 19.289 | 18.370 | 0.990 | 0.943 |
| v1 | FFT | 16 | 20.181 | 18.920 | 19.174 | 0.938 | 0.950 |
| v1 | SHT | 1 | 10.318 | 10.545 | 10.242 | 1.022 | 0.993 |
| v1 | SHT | 16 | 10.205 | 10.080 | 10.169 | 0.988 | 0.996 |
| v1 | NUFFT | 1 | 10.306 | 10.482 | 10.518 | 1.017 | 1.021 |
| v1 | NUFFT | 16 | 10.411 | 10.274 | 10.465 | 0.987 | 1.005 |
| v1 | wgridder | 1 | 134.277 | 139.816 | 135.155 | 1.041 | 1.007 |
| v1 | wgridder | 16 | 141.832 | 154.618 | 132.042 | 1.090 | 0.931 |
| v3 | FFT | 1 | 13.401 | 13.121 | 13.155 | 0.979 | 0.982 |
| v3 | FFT | 16 | 13.212 | 13.095 | 13.295 | 0.991 | 1.006 |
| v3 | SHT | 1 | 7.369 | 7.556 | 7.514 | 1.025 | 1.020 |
| v3 | SHT | 16 | 7.447 | 7.266 | 7.063 | 0.976 | 0.948 |
| v3 | NUFFT | 1 | 6.873 | 6.804 | 6.819 | 0.990 | 0.992 |
| v3 | NUFFT | 16 | 6.748 | 6.843 | 6.987 | 1.014 | 1.035 |
| v3 | wgridder | 1 | 97.514 | 97.336 | 95.830 | 0.998 | 0.983 |
| v3 | wgridder | 16 | 97.350 | 97.757 | 97.503 | 1.004 | 1.002 |

Some broad-matrix rows vary materially across runs; for example, the B v1 16-thread wgridder row has 7.6% CV. The focused FFT comparison is the cleaner read for the targeted FFT question. The latest broad matrix also does not show the previous multiarch B FFT slowdown.

Earlier focused comparisons did show B slower: B/A was 1.171, 1.158, 1.139, and 1.122 for v1/1T, v1/16T, v3/1T, and v3/16T. The earlier full matrix showed B/A of 1.088, 1.156, 1.184, and 1.120 on those same FFT cases. The largest earlier values are not reproduced in the latest explicit-environment runs. The source of the difference is unknown; benchmark-worker environment handling differed between generations, so the earlier numbers should remain visible but not drive a claim of a stable regression.

The complete raw measurements and CVs are in the local `followup-benchmark-summary.json`; raw pyperf files are under `/tmp/ducc-lto-results/A`, `/tmp/ducc-lto-results/B`, and `/tmp/ducc-lto-results/B-FFT-noLTO-release-sampled`. These machine-readable run artifacts are not included in the Markdown-only documentation commit.

## Native LTO ablation and code-generation evidence

A separate non-multiarch build pair disabled IPO only for the `${PKGNAME}_lib` target in the noLTO case; the final Python module target retained its normal link behavior. In five-process focused repeats, noLTO was faster for both thread settings:

| Threads | Native LTO | Native noLTO | noLTO/LTO | noLTO improvement |
|---:|---:|---:|---:|---:|
| 1 | 14.968 ms | 13.397 ms | 0.895 | 10.5% |
| 16 | 14.858 ms | 13.521 ms | 0.910 | 9.0% |

The effect depended on shape. On the expanded three-shape run, noLTO was 5.3% faster for complex64 `17x120x196`, 10.6% faster for real64 `8x255x257`, and 0.8% slower for real32 `8x256x256`. This supports a shape-dependent native LTO codegen effect, not a universal FFT slowdown.

Two evidence sources point to inlining as one mechanism:

- GCC 15's B LTO optimization report records 246 `exec_ -> special_mul` missed-inline call sites, all attributed to `--param inline-unit-growth limit reached`; it contains no successful `special_mul` inline lines. The report is 40.9 MB.
- `py-spy` sampled the native LTO and noLTO binaries for about 11k samples each. `ducc0::detail_fft::c2c<double>` accounts for about 87–88% inclusive frames. `special_mul<true/false, SIMD<double, VecBuiltin<32>>,double>` appears in about 16.7% of LTO stacks and has no corresponding noLTO hot frames. A/B profiles also show differing time in FFT execution and copy/input paths.

The compiler report and samples support a codegen explanation but do not quantify how much of the end-to-end difference is caused by this helper. Hardware counters were unavailable, so there are no cycles, instruction counts, IPC, or cache measurements.

## ISA boundary audit

ISA counts are from decoded instructions grouped by profile or the residual `other` bucket. “Other” contains startup/dispatcher and other non-profile symbols. These are static audits; the host did not execute v4 code.

### Audit method and reproduction

The audit implementation used for these results is `/tmp/ducc-lto-results/audit-isa.py` (kept in the local analysis artifacts, not in this repository). For an already-built extension, run:

```sh
python3 /tmp/ducc-lto-results/audit-isa.py /path/to/ducc0.cpython-313-x86_64-linux-gnu.so /tmp/isa-inventory.json
```

It runs `nm -C -S --defined-only` to collect defined text symbols and their address ranges, then `objdump -d -C -M intel --insn-width=16` to decode the final ELF. Each decoded instruction address is attributed to every covering function range; demangled `ducc0_v1`, `ducc0_v3`, and `ducc0_v4` names identify the profile buckets, with 32-byte/64-byte libstdc++ SIMD helpers attributed by their ABI type and all remaining code placed in `other`. It classifies legacy/VEX/EVEX from the instruction prefix bytes and flags VEX/EVEX, selected v2/v3 instructions, and YMM/ZMM/opmask operands where those exceed a bucket's target ISA. The 16-byte disassembly width preserves complete x86 instruction bytes in the JSON.

For example, the v4/v3/v1 GCC 15 link-order ELF has an `other`-bucket EVEX at `0x281730`: `62 f1 fd 48 6f 05 46 07 09 02`, `vmovdqa64 zmm0,...`, in `std::__cxx11::to_string(long)`. It can be independently inspected with:

```sh
objdump -d -C -M intel --insn-width=16 --start-address=0x281730 --stop-address=0x28173a /tmp/ducc-lto-results/GCC15-link-order-v4-v3-v1/ducc0.cpython-313-x86_64-linux-gnu.so
```

The order comparison is generated by `/tmp/ducc-lto-results/compare-linkorder-inventories.py` and saved as `/tmp/ducc-lto-results/GCC15-link-order-comparison.json`. Link commands are saved in each order's `link-command.txt`; they use the same GCC flags and 28 objects per profile, with only the v1/v3/v4 profile-object sequence changed. These scripts, inventories, and ELF files remain in the local `/tmp/ducc-lto-results` analysis directory and are not part of this Markdown-only commit. This is reproducible static evidence for the inspected ELFs, not a formal proof for all compiler versions or a runtime test on x86-64-v1 hardware.

### GCC 15 B, default profile link order

The default order was `v1, v3, v4`. It passed the audit: v1 contains legacy instructions only, v3 contains VEX but no EVEX, v4 contains EVEX, and `other` contains legacy instructions only. This is an empirical pass for GCC 15.2 and this exact order, not a compiler guarantee.

### GCC 14 B

The named v1/v3/v4 profile namespaces pass their expected instruction boundaries, but `other` contains 12 VEX instructions in `_sub_I_65535_0.0`. The `.init_array` startup constructor initializes the v1/v3/v4 `KernelDB` vectors from `src/ducc0/math/gridding_kernel.cc:28`; those VEX instructions can run before profile dispatch. Forced v1 smoke tests on this AVX2 host pass but cannot establish baseline safety on an x86-64-v1 machine.

### GCC 15 link-order stress

Only profile objects were manually reordered in the saved default link command. Results:

| Profile object order | VEX in `other` | EVEX in `other` | Static interpretation |
|---|---:|---:|---|
| v1, v3, v4 (default) | 0 | 0 | baseline clean |
| v4, v3, v1 | 4,633 | 594 | common pybind dispatcher/call and `std::to_string` code contains higher ISA |
| v3, v1, v4 | 4,056 | 0 | common code contains VEX; root `PyInit`/`add_cpu_info` symbols remain legacy |

Decoded instruction counts by profile namespace and order are below. `L/V/E` means legacy/VEX/EVEX; the v1 and v3 rows contain no EVEX in any order.

| Profile bucket | v1, v3, v4 (default) L/V/E | v4, v3, v1 L/V/E | v3, v1, v4 L/V/E |
|---|---:|---:|---:|
| v1 | 2,364,066 / 0 / 0 | 2,318,492 / 0 / 0 | 2,289,804 / 0 / 0 |
| v3 | 1,870,607 / 525,470 / 0 | 1,812,726 / 538,263 / 0 | 1,905,099 / 530,694 / 0 |
| v4 | 1,901,611 / 408,635 / 180,163 | 1,999,674 / 425,120 / 184,393 | 1,930,988 / 412,051 / 180,972 |
| other | 88,293 / 0 / 0 | 119,816 / 4,633 / 594 | 105,835 / 4,056 / 0 |

The first reordered build also increases the stripped wheel/extension/`.text` sizes to 13,626,419 / 40,437,616 / 35,114,944 B; the second yields 13,519,057 / 40,187,728 / 34,850,288 B. Default-order B is 13,447,156 / 40,089,360 / 34,752,272 B. This demonstrates that the LTO boundary depends on link order; passing one order is insufficient for a robust guarantee.

### Clang 21 A and B

Both A and B pass their named v1/v3/v4 profile namespace checks. Both have four VEX instructions in `other`, in `_GLOBAL__sub_I_gridding_kernel.cc`, which initializes the same kernel tables during startup. So both Clang builds have a baseline startup ISA violation. Forced v1/v3 numerical smoke tests passed on this AVX2 host, but the full Python/performance matrix was not run because the static baseline audit already found this issue. The Clang C++ driver passed 12/12.

For context, stripped release-size proxies from the O3 `-g` debug builds are 9,235,869 B (A) and 9,021,017 B (B) for the wheel; the extensions are 34,333,264 B and 32,106,448 B. These are proxies, not production release builds.

### FFT-noLTO hybrid

The hybrid's raw GCC 15 `other` bucket contains 20 VEX instructions. The corrected object/final-symbol audit assigns all 20 to four local `std::__introsort_loop` helper clones in the non-LTO FFT instantiation objects for v3 and v4. They have local (`t`) binding and remain confined to their source objects. The v1 helper copies are legacy-only; the v3/v4 helpers contain the expected VEX instructions. No other `other` symbol is flagged. This classification depends on the recorded object list, symbol bindings, link order, and final addresses; the hybrid has not received the GCC 14, Clang, or link-order stress audit.

The full inventories and corrected helper analysis are in `/tmp/ducc-lto-results/{B-debug,GCC14-B-debug,Clang21-A-debug,Clang21-B-debug,B-FFT-noLTO-debug,GCC15-link-order-v*-v*-v*}`. The hybrid summary is in the local `B-FFT-noLTO-debug/isa-interpretation.json` artifact.

## State and artifacts

The PR worktree and all experimental worktrees were clean after the experiments. No PR code branch was changed or pushed as part of this follow-up; this report is being published separately on `docs/lto-isa-audit-evidence`.

| Worktree | Branch | Commit |
|---|---|---|
| PR | `test/multiarch-dispatch` | `e41fa307d45b` |
| A | `exp/multiarch-lto-a` | `e41fa307d45b` |
| B | `exp/multiarch-lto-b` | `0e3edd3e28b3` |
| D | `exp/multiarch-lto-d` | `d9c0eec650b7` |
| FFT-noLTO hybrid | `exp/lto-followup-b-fft-nolto` | `4fa0090c9ff6` |
| Native noLTO ablation | `exp/lto-followup-native-nolto` | `cd7585ea6f45` |

Local analysis artifacts: `followup-results.json`, `followup-benchmark-summary.json`, and the earlier `final-report.md` are under `/tmp/ducc-lto-results`; the checked-in Markdown contains the relevant summarized findings and methodology.

## Follow-up evidence and required answers

### Source revision and reproducibility

Repository: https://github.com/mwyau/ducc. The pushed PR head is **e41fa307d45b9df465f3429587658e37f3db3b4e**; upstream/multiarch is **8aabaa0e925812a440f414e1df5cb817c4b70aa5**, and the merge base is exactly that upstream SHA. The PR worktree at **/home/albert/ducc-dispatch-tests** is clean. Experiments used separate worktrees; no PR branch edits, pushes, merges, or GitHub comments were made.

| Variant | Worktree | Branch | Commit |
|---|---|---|---|
| PR / A | /home/albert/ducc-dispatch-tests / /home/albert/ducc-lto-a | test/multiarch-dispatch / exp/multiarch-lto-a | e41fa307d45b9df465f3429587658e37f3db3b4e |
| B | /home/albert/ducc-lto-b | exp/multiarch-lto-b | 0e3edd3e28b32b6444825c07b5c96f5a281905bb |
| D | /home/albert/ducc-lto-d | exp/multiarch-lto-d | d9c0eec650b7a34556b9fe02f00620709cadc48f |
| B with FFT instantiations noLTO | /home/albert/ducc-b-fft-nolto | exp/lto-followup-b-fft-nolto | 4fa0090c9ff63ab083291b5563f1901a9615835f |
| Native noLTO control | /home/albert/ducc-native-nolto | exp/lto-followup-native-nolto | cd7585ea6f458136192604f9cdd9459731464bbb |

Saved command records include A-debug/all-commands.txt, B-debug/all-commands.txt, their compile databases, B-debug/lto-report/link-command.shlex.json, the hybrid final link command, GCC 14 and Clang compile databases, and the three link-order link commands. GCC 15 B profile compilation contains -flto=auto and -fno-fat-lto-objects; A profile compilation does not. In the hybrid only fft_inst1.cc and fft_inst2.cc lose LTO in v1/v3/v4; other profile objects retain LTO.

Tool versions: GCC/G++ 15.2.0, GCC/G++ 14.3.0, Clang 21.1.8 with LLD 21.1.8, GNU binutils 2.46, CMake 4.4.4, Ninja 1.13.2, Python 3.13.13, NumPy 2.5.3, pybind11 3.1.0, pyperf 2.10.0. Clang B uses ThinLTO with -flto=thin and LLD with -fuse-ld=lld; Clang A profile objects omit LTO but the final driver uses ThinLTO.

Host: AMD Ryzen 9 5950X, 16 cores / 32 threads, one NUMA node. It supports v1 and v3, not v4. Benchmarks used CPUs 16–31, performance governor and EPP, seed 77631, pyperf --copy-env, and one-thread external BLAS/OpenMP pools. Focused pyperf runs used five processes and seven measured values per process; the broad A/B matrix used three measured worker runs with three values each. The native eight-workload matrix used three measured worker runs, one warmup, and min_time=0.1. No v4 instruction was executed.

### Native LTO control: all workloads and FFT shapes

The native noLTO build disabled IPO only for the DUCC library target; the final module target retained its normal setting. Focused results have 35 measured values each:

| Workload | Threads | Native LTO | Native noLTO | noLTO / LTO |
|---|---:|---:|---:|---:|
| FFT focused | 1 | 14.968 ms | 13.397 ms | 0.895 |
| FFT focused | 16 | 14.858 ms | 13.521 ms | 0.910 |

Expanded FFT shapes:

| FFT case | Native LTO | Native noLTO | noLTO / LTO |
|---|---:|---:|---:|
| complex64, 17×120×196, mixed/awkward | 1.507 ms | 1.427 ms | 0.947 |
| real32, 8×256×256, power of two | 0.782 ms | 0.789 ms | 1.008 |
| real64, 8×255×257, awkward | 5.013 ms | 4.482 ms | 0.894 |

Eight-workload matrix. CV is across measured pyperf values; ratios are native noLTO/native LTO. Higher-CV rows are directional.

| Workload | Threads | LTO (ms; CV) | noLTO (ms; CV) | noLTO / LTO |
|---|---:|---:|---:|---:|
| FFT | 1 | 14.788; 8.0% | 13.842; 4.2% | 0.936 |
| FFT | 16 | 15.410; 5.2% | 13.170; 5.5% | 0.855 |
| SHT | 1 | 7.724; 1.3% | 7.784; 1.4% | 1.008 |
| SHT | 16 | 7.616; 1.8% | 7.690; 2.6% | 1.010 |
| NUFFT | 1 | 7.270; 1.4% | 7.142; 1.6% | 0.982 |
| NUFFT | 16 | 7.055; 2.3% | 6.968; 10.0% | 0.988 |
| wgridder | 1 | 106.966; 1.1% | 109.049; 6.3% | 1.019 |
| wgridder | 16 | 100.574; 3.6% | 111.128; 5.3% | 1.105 |

This establishes that LTO can produce a meaningful FFT regression in native builds without multiarch dispatch. The matrix does not show the same effect in SHT or NUFFT, and does not support a general wgridder claim.

### A/B replication and the noisy crossover

The earlier saved result showed B slower than A by 12–17% in focused FFT and 9–18% in broad FFT rows. That did not reproduce in the later same-session --copy-env run: focused B/A medians were 1.002, 1.015, 1.005, and 1.006 for v1/1T, v1/16T, v3/1T, and v3/16T. The latest separate broad matrix ranged from B slightly faster to 9% slower depending on workload, with no repeatable FFT penalty.

An ABBA/BAAB release-wheel crossover asserted the active profile inside every worker. Pooled B/A medians were 1.152, 1.044, 1.148, and 1.113 in that same order. An unrelated CPU-intensive process overlapped the run on CPUs 0–31; load was elevated, and one B block reached 8.7% CV. This is a contaminated observation, not a clean confirmation. The historical B focused run also contains a worker with 59 runnable threads and high CV. No new timing run was started while unrelated CPU load was active. The gap between benchmark generations remains unexplained.

Thus the multiarch B runtime comparison is less certain than the native ablation: the original B/A penalty is not established as stable, while native noLTO was faster on the focused shapes and two of three expanded shapes.

### Profiling, counters, and code generation

Hardware counters could not be read. perf stat for cycles, instructions, branches, branch misses, cache references, and cache misses returned “No supported events found”; /proc/sys/kernel/perf_event_paranoid is 4. sudo -n could not acquire credentials. No sysctl was changed. Therefore there is no cycles, instructions, IPC, branch, or cache comparison, and perf record/report/annotate produced no samples. See perf-stat-smoke-result.json and perf-stat-smoke.stderr.txt.

py-spy recorded 11,236 samples for native LTO and 11,599 for native noLTO. detail_fft::c2c<double> appeared in about 87–88% of inclusive frames in both; the special_mul SIMD32 helper appeared in about 16.7% of LTO stacks and not among noLTO hot frames. A/B sample totals were 12,332 (A) and 11,555 (B); c2c<double> accounted for 80.46% and 86.13% of inclusive samples, and ExecC2C::exec_n for 42.90% and 48.24%. These percentages describe sampled stack presence, not absolute time or speedup. Raw profiles are native-lto-debug/fft-profile.raw, native-no-lto-debug/fft-profile.raw, and A-B-hot-profile.json.

The targeted O3 -g disassembly uses the same v1 cfftp8<double>::exec_ SIMD16 kernel in each binary. A has 1,086 instructions and no special_mul calls; B has 1,041 instructions and eight calls, seven to special_mul; native LTO has 847 instructions and seven such calls; native noLTO has 834 instructions and none. B has 135 stack-memory instructions versus A's 108; native LTO/noLTO have 138 versus 114. This confirms a concrete call/inlining difference in one relevant kernel, but does not prove the end-to-end mechanism. See fft-codegen-comparison.json and the assembly files in the debug build directories.

The GCC 15 LTO optimization report contains 246 missed exec_ to special_mul inlining decisions. Every recorded reason is --param inline-unit-growth limit reached; there are zero successful special_mul inline lines. The report is 40,875,666 bytes. This is consistent with extra out-of-line calls; their contribution to total runtime is not quantified. See B-debug/lto-report/inline-optimized-missed.txt, B-debug/lto-report/special-mul-summary.json, and B-debug/lto-dump-fft-inst1.txt.

### Namespace-wide ISA audit and compiler matrix

The audit enumerates defined text/function symbols with nm -C -S --defined-only, disassembles with GNU objdump -d -C -M intel, then assigns every decoded instruction address to each defined symbol range covering it. Suspicious records contain bytes, mnemonic, operands, encoding, minimum-feature flags, symbol, and address. The legacy-encoded x86-64-v2 scan includes SSE3/SSSE3/SSE4 instructions, POPCNT, CMPXCHG16B, LAHF/SAHF, plus separate BMI/LZCNT, MOVBE, VEX/EVEX, and vector-register checks. A follow-up scan found no movntdqa, MOVBE, POPCNT, AES, PCLMUL, SHA, GFNI, or VAES in any audited v1 inventory. This goes beyond YMM/ZMM grep. See audit-isa.py, isa-v1-extra-feature-scan.json, and each build's isa-inventory.json.

| Build | v1 namespace | v3 namespace | v4 namespace | Common/startup (other) | Result |
|---|---|---|---|---|---|
| GCC 15 B, v1/v3/v4 order | 2,364,066 legacy instructions; clean | 2,396,077 instructions, 525,470 VEX, no EVEX | 2,490,409 instructions, 180,163 EVEX | 88,293 legacy only | Pass for this compiler and order |
| GCC 14 B | namespace clean; 2,424,060 v1 instructions | clean of v4 | EVEX as expected | 12 VEX in _sub_I_65535_0.0 startup constructor | Baseline startup violation |
| Clang 21 A | namespace clean | clean of EVEX | EVEX present | 4 VEX in _GLOBAL__sub_I_gridding_kernel.cc | Baseline startup violation |
| Clang 21 B, ThinLTO | namespace clean | clean of EVEX | EVEX present | 4 VEX in _GLOBAL__sub_I_gridding_kernel.cc | Baseline startup violation |
| GCC 15 B-FFT-noLTO, default order | clean | clean of EVEX | EVEX present | 20 raw VEX, all classified to four local std::__introsort_loop copies in v3/v4 FFT objects | Conditional pass after attribution |

GCC 14's 12 VEX instructions are in _sub_I_65535_0.0, whose constructor initializes kernel database vectors from src/ducc0/math/gridding_kernel.cc. Clang A and B each have four VEX instructions in the startup global initializer for that translation unit. These are import/startup paths before profile dispatch. Full inventories contain exact instruction bytes, addresses, mnemonics, operands, and encodings. GCC 15 B default has no suspicious common/startup instruction.

For GCC 15 link-order stress only profile-object order changed. v1/v3/v4 profile namespaces stayed within their assigned ISA in all three builds: v1 remained legacy-only, v3 had VEX but no EVEX, and v4 contained EVEX. The common bucket changed: default order had no VEX/EVEX; v4/v3/v1 had 4,633 VEX and 594 EVEX; v3/v1/v4 had 4,056 VEX. Thus these tests did not show cross-contamination inside profile namespaces, but common/startup code is link-order fragile. Exact records and commands are in the GCC15 link-order build directories. Wheel / extension / .text sizes were 13,447,156 / 40,089,360 / 34,752,272 B by default; 13,626,419 / 40,437,616 / 35,114,944 B for v4/v3/v1; 13,519,057 / 40,187,728 / 34,850,288 B for v3/v1/v4.

The hybrid's 20 raw other-bucket VEX instructions all belong to four local std::__introsort_loop copies in v3/v4 non-LTO FFT objects. Object attribution and final symbol addresses show they stay with those profile objects; v1 copies are legacy-only. No other other-bucket symbol is flagged. This is only a GCC 15 default-order result; no GCC 14, Clang, or link-order audit was done for the hybrid.

GCC 14's O3 -g build took 2:27.65. Its stripped/repacked size proxy was a 13,511,546 B wheel, 40,728,336 B extension, and 35,245,696 B .text; this proxy comes from a debug build, not a release wheel. Clang A/B debug builds took 90.61 s and 120.01 s. Their stripped/repacked size proxies:

| Clang build | Wheel proxy | Extension proxy | .text |
|---|---:|---:|---:|
| A | 9,235,869 B | 34,333,264 B | 28,542,625 B |
| B | 9,021,017 B | 32,106,448 B | 27,507,633 B |

Both Clang variants imported and reported cpu_info; forced v1/v3 numerical smoke checks passed on this AVX2 host. Clang A's C++ driver passed 12/12. Clang B's C++ driver and both full Python suites were not run after finding the static startup violation. No Clang performance matrix was run for the same reason. GCC 14 B passed import/cpu_info and forced v1/v3 numerical smoke, but host checks do not establish safety on a v1-only CPU. Its full suite was not run.

### Correctness and remaining scope

| Variant | C++ driver | Full Python suite | Forced v1 / v3 | Static v4 | Notes |
|---|---:|---:|---|---|---|
| GCC 15 A, B, D | 12/12 each | 57,507 passed; 4,632 skipped each | v1 matched A bitwise; v3 within configured tolerance | Audited, not executed | Tests do not prove v1 ISA safety |
| GCC 15 B-FFT-noLTO | 12/12 | 57,507 passed; 4,632 skipped | v1 matched A bitwise; v3 within configured tolerance | Audited, not executed | Audit only GCC 15 default order |
| GCC 14 B | Not run | Not run | Import and numerical smoke passed | Audited; startup VEX found | Not safe to recommend |
| Clang 21 A | 12/12 | Not run | Import and numerical smoke passed | Audited; startup VEX found | No universal baseline proof |
| Clang 21 B | Not run | Not run | Import and numerical smoke passed | Audited; startup VEX found | ThinLTO; not a performance candidate |
| GCC 15 reordered B links | Not run | Not run | Import and numerical smoke passed for v1/v3 | Audited | Profile buckets pass; other fails in two orders |

No nanobind behavior was changed. v4 was inspected statically only; this host lacks AVX512. A Python test pass is not evidence of v1 ISA safety.

For an AVX512-capable runner, from the selected candidate checkout:

~~~~sh
DUCC0_OPTIMIZATION=multiarch-release python -m pip wheel . --no-deps --no-build-isolation -w dist \
  -Cbuild-dir=build-avx512 \
  -Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
python -m pip install --force-reinstall --no-deps dist/ducc0-*.whl
DUCC0_MAX_PSABI_LEVEL=4 python -c 'import ducc0; i=ducc0.misc.cpu_info(); assert i["active_profile"] == "x86-64-v4", i; print(i)'
DUCC0_MAX_PSABI_LEVEL=4 python -m pytest python/test
~~~~

Use the same fixed inputs and thread-pool settings for the optional focused benchmark. Do not run this command on the current host.

### Final comparison and six answers

| Variant | FFT vs A | Other runtime | Clean build | Wheel | ISA confidence | Complexity |
|---|---|---|---:|---:|---|---|
| A | Baseline | Baseline | 114.2 s median | 14.78 MB | GCC 15 default clean; Clang A has startup VEX | Lowest |
| B | Latest clean focused run near A; earlier +12–17% not reproduced | Generally near A with noisy per-case variation | 91.2 s median | 13.45 MB | Not robust: GCC 14, Clang 21, and two GCC 15 orders leak into startup/common code | Low |
| B-FFT-noLTO | Within 0.0–1.6% of A in focused run | Broad run varies by workload | 91.82 s, one run | 13.13 MB | GCC 15 default order passes after helper attribution; other compiler/order work remains | Medium |
| D | Historical B/D FFT about 12–16% slower than A | Mostly near A | 293.5 s median | 13.39 MB | Profile-local strategy; no new cross-compiler/order evidence | High |
| Clang B | Not measured | Not measured | 120.01 s O3 -g build | 9.02 MB proxy | Startup baseline audit fails; ThinLTO | Low |

#### 1. Is the FFT slowdown caused by LTO itself?

Yes, LTO itself can slow DUCC FFT: native noLTO improved focused 1T and 16T by 10.5% and 9.0%, and improved two of three expanded shapes by 5.3% and 10.6%. One expanded shape was 0.8% slower without LTO. The missed-inline report and disassembly support lost special_mul inlining as one plausible mechanism, but do not quantify its contribution. The original large B/A penalty remains inconclusive: clean reruns were near parity, while the strongest interleaved confirmation overlapped unrelated CPU load.

#### 2. Does B preserve ISA isolation?

- **GCC 15.2, default order:** yes in this audited binary; profile namespaces meet boundaries and other is baseline-only.
- **GCC 14.3:** no; startup kernel-table constructor contains 12 VEX instructions.
- **GCC 15 link-order permutations:** profile namespaces remain clean, but other contains 4,633 VEX + 594 EVEX in v4/v3/v1 order and 4,056 VEX in v3/v1/v4.
- **Clang 21.1.8 / ThinLTO B:** no; Clang A and B both have four VEX instructions in the startup kernel-table initializer.

These are empirical results, not a documented compiler guarantee. Machine-readable audits retain exact suspicious instruction locations.

#### 3. Is B robust enough to ship?

No. One GCC 15 default-order pass is outweighed by GCC 14, Clang, and GCC 15 order-stress startup findings. The profile namespaces passed, but import-time/common code must also remain baseline-safe.

#### 4. Does FFT-only noLTO fix B's major downside?

It removes nearly all of the focused A/B gap in the measured GCC 15 run: hybrid/A was 1.000, 1.016, 1.011, and 1.003 for v1/1T, v1/16T, v3/1T, and v3/16T. Its one clean build was about 19.6% faster than A and near B; its wheel was 11.2% smaller than A and 2.4% smaller than B. Extension and .text were each about 0.3% smaller than B. It is a credible experiment, not yet a shipping candidate, because compiler/link-order audits are incomplete and the strongest B crossover was contaminated.

#### 5. What should PR #80 do?

Keep A on PR #80. The clean PR branch was not changed. Revisit B or the hybrid after the common startup initializer and link-order boundary are robust across supported compilers.

#### 6. Is C still worth implementing?

No, not as the next step. Native testing shows that keeping FFT instantiations out of LTO can recover performance; it does not show that a second shared object is needed. The tested hybrid creates that boundary with two translation units while retaining the rest of B's LTO. A shared-object design adds packaging and ABI complexity and only helps if it keeps FFT units outside the LTO domain. If FFT stays in LTO, it could reproduce the same code-generation behavior. C was not implemented.

Machine-readable benchmark rollup: **followup-benchmark-summary.json**. Machine-readable build, ISA, correctness, toolchain, and limitation data: **followup-results.json**. Raw timings remain in pyperf JSON files indexed by the benchmark rollup.
