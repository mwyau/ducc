import json
import os
import subprocess
import sys

import ducc0
import pytest

LEVELS = ("x86-64", "x86-64-v2", "x86-64-v3", "x86-64-v4")
FEATURES = ("sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx",
            "avx2", "avx512")
pytestmark = pytest.mark.skipif(
    not hasattr(ducc0.misc, "cpu_info"),
    reason="runtime CPU dispatch is not enabled in this build")


def _fresh_info(cap):
    env = os.environ.copy()
    env["DUCC0_CPU_MAX"] = cap
    code = "import json,ducc0; print(json.dumps(ducc0.misc.cpu_info()))"
    return subprocess.run([sys.executable, "-c", code], env=env, check=True,
                          capture_output=True, text=True)


def _expected(compiled, usable, cap):
    limit = min(LEVELS.index(usable), LEVELS.index(cap))
    return max((level for level in compiled if LEVELS.index(level) <= limit),
               key=LEVELS.index)


def test_cpu_info_is_compact_and_matches_compiled_profiles():
    info = ducc0.misc.cpu_info()
    assert set(info) == {"architecture", "features", "usable_level",
                         "selection_cap", "fft"}
    assert info["architecture"] == "x86-64"
    assert info["usable_level"] in LEVELS
    assert info["selection_cap"] in LEVELS
    assert info["features"] == [f for f in FEATURES if f in info["features"]]
    assert info["features"][0] == "sse2"
    assert ("avx512" in info["features"]) == (
        info["usable_level"] == "x86-64-v4")
    fft = info["fft"]
    assert fft["compiled"][0] == "x86-64"
    assert set(fft["compiled"]) <= set(LEVELS)
    assert fft["selected"] in fft["compiled"]
    assert fft["selected"] == _expected(
        fft["compiled"], info["usable_level"], info["selection_cap"])


def test_fresh_process_cap_and_fallback():
    compiled = ducc0.misc.cpu_info()["fft"]["compiled"]
    for cap in LEVELS:
        info = json.loads(_fresh_info(cap).stdout)
        assert info["selection_cap"] == cap
        assert info["fft"]["selected"] == _expected(
            compiled, info["usable_level"], cap)


def test_invalid_cap_falls_back_to_v1():
    result = _fresh_info("avx2")
    info = json.loads(result.stdout)
    assert info["selection_cap"] == "x86-64"
    assert info["fft"]["selected"] == "x86-64"
    assert "DUCC0_CPU_MAX must be" in result.stderr
