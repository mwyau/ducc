#!/usr/bin/env bash
# Build and run the maintained C++ tests from the repository root.
set -uo pipefail

cd "$(dirname "$0")/.."

read -r -a cxx <<< "${CXX:-c++}"
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ducc-cpp-tests.XXXXXX") || exit 1
trap 'rm -rf "$tmpdir"' EXIT

common_flags=(-std=c++17 -O1 -g -Isrc -Ipython -DDUCC0_NAMESPACE=ducc0)
regression_sources=(
  src/ducc0/healpix/healpix_base.cc
  src/ducc0/healpix/healpix_tables.cc
  src/ducc0/infra/mav.cc
  src/ducc0/infra/threading.cc
  src/ducc0/math/gl_integrator.cc
  src/ducc0/math/space_filling.cc
  src/ducc0/math/pointing.cc
  src/ducc0/math/geom_utils.cc
  src/ducc0/infra/string_utils.cc
  src/ducc0/infra/system.cc
)

passed=0
failed=0

run_step() {
  local name=$1
  shift
  printf 'RUN   %s\n' "$name"
  if "$@"; then
    printf 'PASS  %s\n' "$name"
    passed=$((passed + 1))
  else
    local status=$?
    printf 'FAIL  %s (exit %s)\n' "$name" "$status"
    failed=$((failed + 1))
  fi
}

run_step "build regression tests (AddressSanitizer)" \
  "${cxx[@]}" "${common_flags[@]}" -fsanitize=address \
  -o "$tmpdir/test_regressions" test/test_regressions.cc \
  "${regression_sources[@]}" -pthread

if [[ -x "$tmpdir/test_regressions" ]]; then
  for name in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
    run_step "regression: $name" env ASAN_OPTIONS=detect_leaks=0 \
      "$tmpdir/test_regressions" "$name"
  done
else
  for name in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
    printf 'NOT RUN regression: %s (regression test build failed)\n' "$name"
  done
fi

run_step "build multiarch selection test" \
  "${cxx[@]}" "${common_flags[@]}" -o "$tmpdir/test_multiarch" \
  test/test_multiarch.cc python/multiarch.cc
if [[ -x "$tmpdir/test_multiarch" ]]; then
  run_step "multiarch profile selection and feature detection" \
    "$tmpdir/test_multiarch"
else
  printf 'NOT RUN multiarch profile selection and feature detection (build failed)\n'
fi

run_step "compile-only public API tests" \
  "${cxx[@]}" "${common_flags[@]}" -c test/test_compile_api.cc \
  -o "$tmpdir/test_compile_api.o"

printf '\nC++ tests: %d passed, %d failed\n' "$passed" "$failed"
(( failed == 0 ))
