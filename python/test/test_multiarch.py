import json
import os
import subprocess
import sys

import ducc0


def _profile_number(profile):
    if profile == "x86-64":
        return 1
    prefix = "x86-64-v"
    assert profile.startswith(prefix), profile
    return int(profile[len(prefix):])


def _arm_profile_number(profile):
    return {"neon": 1, "sve": 2, "sve2": 3}[profile]


def test_cpu_info_metadata_and_selector():
    assert "features" in (ducc0.misc.cpu_info.__doc__ or "")
    info = ducc0.misc.cpu_info()
    assert isinstance(info.get("architecture"), str)
    assert info["architecture"]
    assert isinstance(info.get("features"), list)
    assert all(isinstance(feature, str) for feature in info["features"])
    assert isinstance(info.get("multiarch"), bool)

    features = info["features"]
    if info["architecture"] == "x86-64":
        if sys.platform.startswith("linux"):
            assert "sse2" in features
        assert "avx2" not in features or "avx" in features
        assert "avx512" not in features or "avx" in features
    elif info["architecture"] == "aarch64":
        assert set(features) <= {"neon", "sve", "sve2"}

    if not info["multiarch"]:
        assert set(info) == {"architecture", "features", "multiarch"}
        assert info["multiarch"] is False
        return

    assert set(info) == {
        "architecture",
        "features",
        "multiarch",
        "compiled_profiles",
        "available_profiles",
        "configured_limit",
        "active_profile",
    }
    compiled = info["compiled_profiles"]
    available = info["available_profiles"]
    if info["architecture"] == "x86-64":
        assert compiled == ["x86-64", "x86-64-v3", "x86-64-v4"]
        assert available
        assert available == compiled[:len(available)]
        assert available[0] == "x86-64"
        limit_number = _profile_number(info["configured_limit"])
        assert 1 <= limit_number <= 4
        expected = next(profile for profile in reversed(available)
                        if _profile_number(profile) <= limit_number)
    else:
        assert info["architecture"] == "aarch64"
        assert compiled and compiled[0] == "neon"
        assert set(compiled) <= {"neon", "sve", "sve2"}
        assert [_arm_profile_number(p) for p in compiled] == sorted(
            _arm_profile_number(p) for p in compiled)
        assert available and available[0] == "neon"
        assert set(available) <= set(compiled)
        assert [_arm_profile_number(p) for p in available] == sorted(
            _arm_profile_number(p) for p in available)
        limit_number = _arm_profile_number(info["configured_limit"])
        expected = next(profile for profile in reversed(available)
                        if _arm_profile_number(profile) <= limit_number)
    assert info["active_profile"] == expected


def test_malformed_configured_limit_fails_clearly():
    info = ducc0.misc.cpu_info()
    if not info["multiarch"]:
        return

    env = os.environ.copy()
    if info["architecture"] == "aarch64":
        env.pop("DUCC0_MAX_PSABI_LEVEL", None)
        env["DUCC0_MAX_ARM_PROFILE"] = "3invalid"
        expected_error = "DUCC0_MAX_ARM_PROFILE"
    else:
        env["DUCC0_MAX_PSABI_LEVEL"] = "3invalid"
        expected_error = "DUCC0_MAX_PSABI_LEVEL"
    result = subprocess.run(
        [sys.executable, "-c", "import ducc0"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert expected_error in result.stderr


def test_arm_maximum_profile_limit():
    info = ducc0.misc.cpu_info()
    if not info["multiarch"] or info["architecture"] != "aarch64":
        return

    for limit in ("neon", "sve", "sve2"):
        env = os.environ.copy()
        env.pop("DUCC0_MAX_PSABI_LEVEL", None)
        env["DUCC0_MAX_ARM_PROFILE"] = limit
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, ducc0; print(json.dumps(ducc0.misc.cpu_info()))"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        limited = json.loads(result.stdout)
        assert limited["configured_limit"] == limit
        selected = [profile for profile in limited["available_profiles"]
                    if _arm_profile_number(profile)
                    <= _arm_profile_number(limit)]
        assert limited["active_profile"] == selected[-1]
