# Linux AArch64 multiarch builds

Set `DUCC0_OPTIMIZATION=multiarch` (or a `multiarch-*` variant) to build one
Python extension containing a baseline NEON implementation and, when accepted
by the selected compiler, separate SVE and SVE2 implementations. The extension
driver is compiled for `-march=armv8-a`; implementation objects use distinct
DUCC namespaces and per-profile target flags. SVE objects use
`-march=armv8-a+sve` and SVE2 objects use `-march=armv8-a+sve2`; both retain
`-msve-vector-bits=scalable`. Starting from the Armv8-A baseline avoids
implicitly enabling unrelated extensions such as LSE atomics. Multiarch objects
and the extension driver are compiled and linked without LTO so target-specific
code cannot be merged into the baseline dispatch path.

NEON is the Linux AArch64 baseline. CMake probes the compiler before enabling
the optional SVE profiles. If an optional target is unsupported, configuration
reports that profile as disabled and keeps the NEON build. If the required
NEON target is unsupported, configuration stops with an actionable error.
`ducc0.misc.cpu_info()` reports only profiles that were built and are usable on
the running process. Runtime feature detection uses Linux `AT_HWCAP` and
`AT_HWCAP2`; inconsistent SVE/SVE2 feature combinations do not enable those
profiles.

`DUCC0_MAX_ARM_PROFILE=neon|sve|sve2` can cap selection at import time. The
default is `sve2`. A limit only reduces the choices: it cannot enable an ISA
that the kernel does not report. Invalid values raise an import-time error.
The selected implementation is fixed when the module is imported.

## SIMD backend and limits

The SVE profiles allow scalable SVE code generation, and GCC emits SVE
instructions in parts of the profile objects. The diagnostic below reports a
different detail: with the Ubuntu GCC 15.2/libstdc++ toolchain,
`std::experimental::native_simd<double>` still uses a two-lane 128-bit NEON ABI
in the SHT inner loop when `__ARM_FEATURE_SVE_BITS=0`. That probe result does
not account for GCC auto-vectorizing other loops to SVE. The SVE2 target also
emits SVE instructions; the probe and disassembly do not by themselves establish
a performance benefit or additional SVE2-specific kernel acceleration.

Build the standalone diagnostic in `test/arm64_simd_probe.cc` with each exact
target to see the selected native SIMD type and lane count. For example:

```sh
aarch64-linux-gnu-g++ -std=c++17 -O3 -march=armv8-a \
  test/arm64_simd_probe.cc -o /tmp/simd-neon
aarch64-linux-gnu-g++ -std=c++17 -O3 -march=armv8-a+sve \
  -msve-vector-bits=scalable test/arm64_simd_probe.cc -o /tmp/simd-sve
aarch64-linux-gnu-g++ -std=c++17 -O3 -march=armv8-a+sve2 \
  -msve-vector-bits=scalable test/arm64_simd_probe.cc -o /tmp/simd-sve2
aarch64-linux-gnu-objdump -d --disassemble=ducc_simd_probe /tmp/simd-sve
```

Inspect the representative operation and the numerical kernels themselves
before attributing SVE acceleration to a build. The probe is diagnostic and is
not linked into the Python extension.

With the Ubuntu AArch64 GCC 15.2/libstdc++ toolchain used for validation,
`native_simd<double>` reports two lanes for all three targets and the probe
operation is `fmla v...2d` in all three builds. The SVE and SVE2 profile objects
also contain compiler-generated scalable SVE instructions in other code. The
selected numerical tests passed under QEMU on NEON, SVE-only, and SVE2 CPUs, but
performance was not benchmarked and native ARM hardware was not available.
