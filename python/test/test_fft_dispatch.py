import json
import os
from pathlib import Path
import subprocess
import sys

import ducc0
import numpy as np
from numpy.testing import assert_allclose
import pytest

LEVELS = ("x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4")
pytestmark = pytest.mark.skipif(
    not hasattr(ducc0.misc, "cpu_info"),
    reason="runtime CPU dispatch is not enabled in this build")


def _write_results(path):
    import ducc0
    import numpy as np

    fft = ducc0.fft
    rng = np.random.default_rng(93457)
    results = {}
    for real_type, complex_type, suffix in (
            (np.float32, np.complex64, "f32"),
            (np.float64, np.complex128, "f64")):
        real = rng.standard_normal((5, 7, 9)).astype(real_type)
        comp = (rng.standard_normal((3, 5, 7))
                + 1j*rng.standard_normal((3, 5, 7))).astype(complex_type)
        spec = fft.r2c(real, axes=(0, 2), inorm=1)
        mut_spec = spec.copy()
        matrix = real[:, :, 0]
        kernel_r = rng.standard_normal(7).astype(real_type)
        kernel_c = (rng.standard_normal(7) + 1j*rng.standard_normal(7)).astype(
            complex_type)
        out_r = np.empty((5, 11), dtype=real_type)
        out_c = np.empty((11,), dtype=complex_type)
        fft.convolve_axis(matrix, out_r, 1, kernel_r)
        fft.convolve_axis(comp[0, 0], out_c, 0, kernel_c)
        results.update({
            suffix+"_c2c": fft.c2c(comp, axes=(0, 2), inorm=1),
            suffix+"_c2c_strided": fft.c2c(comp.reshape(-1)[::3], inorm=1),
            suffix+"_r2c": spec,
            suffix+"_c2r": fft.c2r(spec, axes=(0, 2), lastsize=9, inorm=2),
            suffix+"_c2r_mut": fft.c2r(
                mut_spec, axes=(0, 2), lastsize=9, inorm=2,
                allow_overwriting_input=True),
            suffix+"_dct": fft.dct(matrix, type=2, axes=(1, 0), inorm=1),
            suffix+"_dst": fft.dst(matrix, type=3, axes=(0, 1), inorm=2),
            suffix+"_fftpack": fft.r2r_fftpack(
                matrix, axes=(1,), real2hermitian=True, forward=True, inorm=1),
            suffix+"_fftw": fft.r2r_fftw(matrix, axes=(1,), forward=False),
            suffix+"_separable_fht": fft.separable_fht(matrix, axes=(0, 1)),
            suffix+"_separable_hartley": fft.separable_hartley(
                matrix, axes=(0, 1)),
            suffix+"_genuine_fht": fft.genuine_fht(matrix, axes=(1, 0)),
            suffix+"_genuine_hartley": fft.genuine_hartley(
                matrix, axes=(1, 0)),
            suffix+"_convolve_real": out_r,
            suffix+"_convolve_complex": out_c,
        })
    np.savez(path, **results)
    print(json.dumps(ducc0.misc.cpu_info()))


def _run_profile(level, path):
    env = os.environ.copy()
    env["DUCC0_CPU_MAX"] = level
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--write", str(path)],
        env=env, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def baseline_output(tmp_path_factory):
    path = tmp_path_factory.mktemp("fft-dispatch") / "v1.npz"
    info = _run_profile("x86-64", path)
    return info, np.load(path)


def test_forced_v1_runs(baseline_output):
    info, output = baseline_output
    assert info["fft"]["selected"] == "x86-64"
    assert output.files


@pytest.mark.parametrize("level", ("x86-64-v3", "x86-64-v4"))
def test_forced_profile_matches_v1(level, baseline_output, tmp_path):
    info = ducc0.misc.cpu_info()
    if level not in info["fft"]["compiled"]:
        pytest.skip(f"{level} is not compiled")
    if LEVELS.index(level) > LEVELS.index(info["usable_level"]):
        pytest.skip(f"{level} is unavailable on this CPU/OS")

    baseline_info, baseline = baseline_output
    assert baseline_info["fft"]["selected"] == "x86-64"
    profile_path = tmp_path / (level.replace("-", "_")+".npz")
    profile_info = _run_profile(level, profile_path)
    assert profile_info["fft"]["selected"] == level
    actual = np.load(profile_path)
    assert actual.files == baseline.files
    for key in baseline.files:
        tol = 2e-6 if actual[key].dtype in (np.float32, np.complex64) else 1e-12
        assert_allclose(ducc0.misc.l2error(actual[key], baseline[key]), 0,
                        atol=tol)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--write":
    _write_results(sys.argv[2])
