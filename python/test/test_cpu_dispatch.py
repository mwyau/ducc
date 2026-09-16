import json
import os
import subprocess
import sys

import numpy as np
import pytest

import ducc0


_TARGETS = ("sse2", "avx", "avx512")
_SUBPROCESS_CODE = r'''
import json
import numpy as np
import ducc0


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def make_alm(lmax, mmax, spin, ncomp, dtype, seed):
    rng = np.random.default_rng(seed)
    result = (rng.standard_normal((ncomp, nalm(lmax, mmax))) +
              1j * rng.standard_normal((ncomp, nalm(lmax, mmax)))).astype(dtype)
    result[:, :lmax + 1].imag = 0
    offset = 0
    for s in range(spin):
        result[:, offset:offset + spin - s] = 0
        offset += lmax + 1 - s
    return result


def encode(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "real": np.asarray(array.real, dtype=np.float64).tolist(),
        "imag": np.asarray(array.imag, dtype=np.float64).tolist(),
    }


def run_case(lmax, mmax, spin, dtype, seed):
    ncomp = 1 if spin == 0 else 2
    ntheta = lmax + 2
    nphi = 2 * mmax + 2
    alm = make_alm(lmax, mmax, spin, ncomp, dtype, seed)
    kwargs = {
        "lmax": lmax,
        "mmax": mmax,
        "spin": spin,
        "nthreads": 1,
        "geometry": "CC",
    }
    synthesis_kwargs = dict(kwargs, ntheta=ntheta, nphi=nphi)
    map_value = ducc0.sht.synthesis_2d(alm=alm, **synthesis_kwargs)
    analysis = ducc0.sht.analysis_2d(map=map_value, **kwargs)
    adjoint_map = ducc0.sht.adjoint_analysis_2d(
        alm=alm, **synthesis_kwargs)
    adjoint_alm = ducc0.sht.adjoint_synthesis_2d(map=map_value, **kwargs)
    return {
        "synthesis": encode(map_value),
        "analysis": encode(analysis),
        "adjoint_analysis": encode(adjoint_map),
        "adjoint_synthesis": encode(adjoint_alm),
    }


info = ducc0.misc.cpu_dispatch_info()
cases = {}
for dtype_name, dtype in (("float32", np.complex64),
                          ("float64", np.complex128)):
    for spin in (0, 1):
        cases["%s_spin%d" % (dtype_name, spin)] = run_case(
            8, 8, spin, dtype, 1000 + 10 * spin +
            (0 if dtype_name == "float32" else 1))
print(json.dumps({"info": info, "cases": cases},
                 sort_keys=True, separators=(",", ":")))
'''


def _run_target(target):
    environment = os.environ.copy()
    environment["DUCC0_CPU_MAX"] = target
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        pytest.fail(
            f"target cap {target} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"target cap {target} did not emit JSON: {exc}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _decode(encoded):
    return (np.asarray(encoded["real"], dtype=np.float64) +
            1j * np.asarray(encoded["imag"], dtype=np.float64))


def test_forced_sht_targets_end_to_end():
    info = ducc0.misc.cpu_dispatch_info()
    if info["mode"] != "dispatch":
        pytest.skip("runtime SHT dispatch is not enabled in this build")

    compiled = set(info["compiled"])
    usable = set(info["usable"])
    assert compiled <= set(_TARGETS)
    assert "sse2" in compiled
    assert "sse2" in usable

    target_results = {}
    skipped = []
    for target in _TARGETS:
        if target not in compiled:
            skipped.append(f"{target}: not compiled")
            continue
        if target not in usable:
            skipped.append(f"{target}: not usable on this runner")
            continue
        print(f"running fresh-process SHT dispatch test for {target}")
        result = _run_target(target)
        assert result["info"]["mode"] == "dispatch"
        assert result["info"]["max"] == target
        assert result["info"]["selected"]["sht.inner"] == target
        target_results[target] = result
    if skipped:
        print("skipped SHT dispatch targets: " + "; ".join(skipped))

    assert "sse2" in target_results
    sse2_cases = target_results["sse2"]["cases"]
    for target, result in target_results.items():
        if target == "sse2":
            continue
        for case_name, reference_case in sse2_cases.items():
            target_case = result["cases"][case_name]
            tolerance = ({"rtol": 5e-5, "atol": 5e-5}
                         if "float32" in case_name
                         else {"rtol": 5e-12, "atol": 5e-12})
            for operation, reference in reference_case.items():
                np.testing.assert_allclose(
                    _decode(target_case[operation]), _decode(reference),
                    err_msg=(f"{target} target differs from sse2 for "
                             f"{case_name}/{operation}"), **tolerance)
