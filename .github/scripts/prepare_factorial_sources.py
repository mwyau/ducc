#!/usr/bin/env python3
"""Create and verify four inline-fix/FFT-tweaks source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BASE_SHA = "d6922b09b85f6c9c20c72cb90a2d6465d219095a"
UPSTREAM_URL = "https://github.com/mreineck/ducc.git"
TWEAK_TIP = "b456d7183ac5e667cff1b769bf5b17e33ebd24cb"
TWEAK_COMMITS = [
    "746ee07d02c01afd68f6faf17a9e9511480e971d",
    "449505438c0fbd1e4544b2ae95653d2b1a150d1b",
    TWEAK_TIP,
]
TWEAK_PATCH_SHA256 = "28137d5773285f100c936ca60d9babe73e36f7857caeeeafdabc55d385e86f86"

CONFIGS = [
    {"id": "A", "special_mul_fix": False, "fft_tweaks": False,
     "label": "Pre-inline-fix · no FFT tweaks"},
    {"id": "B", "special_mul_fix": True, "fft_tweaks": False,
     "label": "Inline fix · no FFT tweaks"},
    {"id": "C", "special_mul_fix": False, "fft_tweaks": True,
     "label": "Pre-inline-fix · FFT tweaks"},
    {"id": "D", "special_mul_fix": True, "fft_tweaks": True,
     "label": "Inline fix · FFT tweaks"},
]

SPECIAL_MUL_PATH = "src/ducc0/fft/fft.h"
TWEAK_PATHS = [
    "src/ducc0/fft/fft1d_impl.h",
    "src/ducc0/fft/fftnd_impl.h",
]
SPECIAL_MUL_OLD = (
    "template<bool fwd, typename T, typename T2> "
    "DUCC0_ALWAYS_INLINE void special_mul "
)
SPECIAL_MUL_OFF = (
    "template<bool fwd, typename T, typename T2> "
    "void special_mul "
)


def run_git(source: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout


def ensure_base_commit(repo_root: Path) -> bool:
    """Ensure the reviewed upstream commit object exists; return whether fetched."""
    exists = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{BASE_SHA}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    fetched = False
    if exists.returncode:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "fetch", "--no-tags", UPSTREAM_URL, BASE_SHA],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"could not fetch pinned upstream commit {BASE_SHA}: "
                f"{result.stderr.strip() or result.stdout.strip()}")
        fetched = True
    verified = run_git(repo_root, "rev-parse", "--verify", f"{BASE_SHA}^{{commit}}").strip()
    if verified != BASE_SHA:
        raise RuntimeError(f"pinned upstream base resolved to unexpected commit {verified}")
    return fetched


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base_file(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE_SHA}:{path}"])


def prepare_variant(source: Path, config: dict, repo_root: Path) -> dict:
    if run_git(source, "rev-parse", "HEAD").strip() != BASE_SHA:
        raise RuntimeError(f"{config['id']}: source worktree is not pinned to {BASE_SHA}")

    tweak_patch = repo_root / ".github/patches/fft_tweaks.patch"
    patch_sha = sha256_bytes(tweak_patch.read_bytes())
    if patch_sha != TWEAK_PATCH_SHA256:
        raise RuntimeError(
            f"FFT tweak patch SHA changed: expected {TWEAK_PATCH_SHA256}, got {patch_sha}")

    if not config["special_mul_fix"]:
        path = source / SPECIAL_MUL_PATH
        contents = path.read_text(encoding="utf-8")
        if contents.count(SPECIAL_MUL_OLD) != 1:
            raise RuntimeError(f"{config['id']}: special_mul patch context is not unique")
        path.write_text(contents.replace(SPECIAL_MUL_OLD, SPECIAL_MUL_OFF, 1),
                        encoding="utf-8")

    if config["fft_tweaks"]:
        subprocess.run(["git", "-C", str(source), "apply", "--check", str(tweak_patch)],
                       check=True)
        subprocess.run(["git", "-C", str(source), "apply", str(tweak_patch)], check=True)

    # Multiarch and native-no-LTO use independent native objects, without IPO.
    # Native LTO builds retain the upstream IPO setting and only one ISA target.
    enable_native_lto = config.get("native_lto", False)
    if not enable_native_lto:
        cmake_path = source / "CMakeLists.txt"
        cmake_text = cmake_path.read_text(encoding="utf-8")
        old_ipo = "set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)"
        if cmake_text.count(old_ipo) != 1:
            raise RuntimeError(f"{config['id']}: unexpected global IPO configuration")
        cmake_path.write_text(
            cmake_text.replace(old_ipo, "set(CMAKE_INTERPROCEDURAL_OPTIMIZATION False)", 1),
            encoding="utf-8")

    source_changes = run_git(source, "diff", "--name-only", "--",
                             "src/ducc0/fft/fft.h",
                             "src/ducc0/fft/fft1d_impl.h",
                             "src/ducc0/fft/fftnd_impl.h").splitlines()
    expected_changes = []
    if not config["special_mul_fix"]:
        expected_changes.append(SPECIAL_MUL_PATH)
    if config["fft_tweaks"]:
        expected_changes.extend(TWEAK_PATHS)
    if sorted(source_changes) != sorted(expected_changes):
        raise RuntimeError(
            f"{config['id']}: unexpected source changes {source_changes}; "
            f"expected {expected_changes}")

    actual_all = run_git(source, "diff", "--name-only").splitlines()
    if sorted(actual_all) != sorted([*expected_changes, *([] if enable_native_lto else ["CMakeLists.txt"])]):
        raise RuntimeError(f"{config['id']}: unexpected worktree changes: {actual_all}")
    subprocess.run(["git", "-C", str(source), "diff", "--check"], check=True)

    inline_bytes = (source / SPECIAL_MUL_PATH).read_bytes()
    expected_inline = base_file(SPECIAL_MUL_PATH)
    if config["special_mul_fix"]:
        if inline_bytes != expected_inline:
            raise RuntimeError(f"{config['id']}: inline-fix ON differs from upstream source")
    else:
        old_bytes = SPECIAL_MUL_OLD.encode("utf-8")
        new_bytes = SPECIAL_MUL_OFF.encode("utf-8")
        if expected_inline.count(old_bytes) != 1:
            raise RuntimeError("pinned base no longer has one special_mul annotation")
        if inline_bytes != expected_inline.replace(old_bytes, new_bytes, 1):
            raise RuntimeError(f"{config['id']}: inline-fix OFF changed more than the annotation")
        numstat = run_git(source, "diff", "--numstat", "--", SPECIAL_MUL_PATH).strip()
        if numstat != f"1\t1\t{SPECIAL_MUL_PATH}":
            raise RuntimeError(f"{config['id']}: special_mul delta is not exactly one line")

    for path in TWEAK_PATHS:
        actual = (source / path).read_bytes()
        expected = base_file(path)
        if not config["fft_tweaks"] and actual != expected:
            raise RuntimeError(f"{config['id']}: FFT tweaks OFF differs from upstream: {path}")
    if config["fft_tweaks"]:
        actual_patch = subprocess.check_output(
            ["git", "-C", str(source), "diff", "--no-ext-diff", "--binary", "--",
             *TWEAK_PATHS])
        if sha256_bytes(actual_patch) != TWEAK_PATCH_SHA256:
            raise RuntimeError(f"{config['id']}: FFT tweak source differs from reviewed patch")

    actual_special_line = next(
        line.strip() for line in (source / SPECIAL_MUL_PATH).read_text().splitlines()
        if "special_mul (" in line)
    return {
        **config,
        "upstream_base_sha": BASE_SHA,
        "special_mul_definition": actual_special_line,
        "source_changes": sorted(source_changes),
        "source_tree_patch_sha256": sha256_bytes(
            subprocess.check_output(["git", "-C", str(source), "diff",
                                     "--no-ext-diff", "--binary", "--",
                                     *expected_changes]) if expected_changes else b""),
        "fft_tweaks_patch_sha256": TWEAK_PATCH_SHA256 if config["fft_tweaks"] else None,
        "fft_tweaks_provenance": {
            "tip": TWEAK_TIP,
            "inspected_commits": TWEAK_COMMITS,
            "files": TWEAK_PATHS,
            "historical_benchmark_commit_excluded": "449505438c0fbd1e4544b2ae95653d2b1a150d1b",
        },
        "benchmark_cmake_patch": (
            "upstream native IPO enabled; single ISA" if enable_native_lto else
            "temporary CMakeLists.txt patch: global IPO disabled"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    worktree_root = args.worktree_root.resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    fetched_base = ensure_base_commit(repo)

    configs = []
    for config in CONFIGS:
        source = worktree_root / config["id"]
        subprocess.run(["git", "worktree", "add", "--detach", str(source), BASE_SHA],
                       cwd=repo, check=True, stdout=subprocess.PIPE)
        configs.append(prepare_variant(source, config, repo))

    manifest = {
        "schema_version": 1,
        "base_upstream": "mreineck/ducc",
        "base_branch": "multiarch",
        "base_sha": BASE_SHA,
        "pinned_base_commit_fetched_from_upstream": fetched_base,
        "source_preparation": "four detached git worktrees from one pinned base commit",
        "factors": ["special_mul_always_inline_fix", "fft_tweaks"],
        "special_mul_factor": {
            "on": "retain the upstream DUCC0_ALWAYS_INLINE annotation",
            "off": "remove only DUCC0_ALWAYS_INLINE on detail_fft::special_mul",
            "portable_macro_changed": False,
        },
        "fft_tweaks": {
            "tip": TWEAK_TIP,
            "inspected_commits": TWEAK_COMMITS,
            "source_patch_sha256": TWEAK_PATCH_SHA256,
            "source_files": TWEAK_PATHS,
            "historical_benchmarking_commit_included": False,
            "namespace_handling_preserved": True,
        },
        "configurations": configs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest),
                      "base_sha": BASE_SHA,
                      "variants": [c["id"] for c in configs],
                      "fft_tweaks_patch_sha256": TWEAK_PATCH_SHA256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
