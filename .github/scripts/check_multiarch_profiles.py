import json
import os

import ducc0


info = ducc0.misc.cpu_info()
print(json.dumps(info, indent=2))

assert info["architecture"] == "x86-64", info
assert info["multiarch"] is True, info
assert info["compiled_profiles"] == [
    "x86-64",
    "x86-64-v3",
    "x86-64-v4",
], info
assert info["available_profiles"], info
assert info["available_profiles"][0] == "x86-64", info
assert info["available_profiles"] == info["compiled_profiles"][:
    len(info["available_profiles"])], info
assert info["configured_limit"] == "x86-64-v4", info
assert info["active_profile"] == info["available_profiles"][-1], info

available = set(info["available_profiles"])
for profile in ("x86-64", "x86-64-v3", "x86-64-v4"):
    if profile in available:
        print(f"AVAILABLE {profile}")
    else:
        print(
            f"SKIP {profile}: cpu_info() reports host-supported profiles "
            f"{info['available_profiles']} and detected features "
            f"{info['features']}"
        )

if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for level, profile in ((3, "x86-64-v3"), (4, "x86-64-v4")):
            print(f"v{level}={'true' if profile in available else 'false'}",
                  file=output)
