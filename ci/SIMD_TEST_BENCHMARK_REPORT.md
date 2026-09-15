# SIMD test and benchmark report

**Validation base:** `d2f74d3803df568b0ce2df6e0094628b5aabe793` (PR #68, `Refactor SSE/AVX/SIMD`)

This branch is a clean validation layer on top of the current PR #68 head. It does not carry the stale history from the earlier public test branch.

## Scope

The production AVX and SSE3 gridding reductions from PR #68 are unchanged. SSE2-only builds continue to use the generic `reduce()` fallback; no explicit SSE2 specialization and no AVX2 backend are added.

The only production experiment on this branch is the opt-in AVX-512 SHT reduction in `src/ducc0/sht/sht_inner_loop.h`:

- candidate A: four `_mm512_reduce_add_pd()` calls;
- candidate B: fold each 512-bit vector to 256 bits, then reuse the existing fused AVX four-stream reduction.

The default AVX-512 path remains the exact PR #68 generic reduction unless a candidate macro is supplied by the validation build.

## Imported exploratory evidence

The latest private validation branch, `mwyau/ducc-private:test-intel-avx512-2`, recorded a follow-up A/B check of the existing AVX float complex horizontal reduction against the alternate shuffle/add sequence.

Using GCC with `-O3 -mavx -mno-avx2`, four randomized A/B blocks, two warmups, five repetitions and 1,000,000 iterations per measurement, the alternate sequence measured:

| Mode | Existing AVX median | Shuffle/add median | Shuffle/add change | Wins |
| --- | ---: | ---: | ---: | ---: |
| Latency | 9.391958 ns | 9.304192 ns | -0.934% | 3/4 |
| Throughput | 3.600835 ns | 3.432082 ns | -4.686% | 4/4 |

For 4,096 deterministic random real/imaginary vector pairs, alternate versus existing AVX had maximum absolute error `2.38419e-07` and maximum relative error `9.45626e-05`. Against a double-precision scalar reference, both had maximum absolute error `1.78814e-07`; the existing AVX path had lower maximum relative error (`1.58768e-05` versus `9.45626e-05`).

These numbers are exploratory only. They do not authorize changing the production AVX implementation, and this branch leaves that implementation unchanged.

## Windows validation

`.github/workflows/tests.yml` contains exactly two visible jobs:

- `Windows 2025 / AVX`
- `Windows 2025 / AVX-512`

Both build the candidate and exact PR #68 baseline with the same clang-cl toolchain and explicit ISA target. The harness uses the MSVC developer environment with the Ninja generator, avoiding assumptions about a particular installed Visual Studio generator version.

The AVX-512 job requires CPU support for AVX512F, DQ, CD, BW and VL plus the required XCR0 state before executing `/arch:AVX512` binaries. If the runner lacks usable AVX-512, it exits successfully with an explicit untested status.

The AVX job also verifies the retained SSE3 `haddps` path and the SSE2-only generic fallback. PR #68 moved `hsum_cmplx` to `detail_gridding_kernel`, so the focused probes on this branch call that current location directly.

For SHT candidate selection, baseline/candidate measurements use four balanced alternating A/B rounds, one thread, deterministic input, warmups, repeated timing and numerical comparison. A candidate is marked as a verified improvement only if aggregate improvement is greater than 1% and it is at least 1% faster in a strict majority of paired rounds.

No AVX-512 candidate should be promoted to production from this branch unless the Windows AVX-512 job actually executes it, passes correctness checks and shows a repeatable end-to-end SHT improvement over the exact `d2f74d3` generic baseline.
