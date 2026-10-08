#!/usr/bin/env python3
"""Build, audit, and measure the eight pinned multiarch FFT configurations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from prepare_factorial_sources import (
    BASE_SHA, CONFIGS, TWEAK_PATCH_SHA256, ensure_base_commit, prepare_variant,
    run_git, sha256_bytes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BENCHMARK_SCRIPT = SCRIPT_DIR / "fft_factorial_benchmark.py"
LTO_HELPER = ROOT / ".github/patches/profile_local_lto.cmake"
PROFILE_LEVELS = {"x86-64": 1, "x86-64-v3": 3, "x86-64-v4": 4}
MARCH = {"x86-64": "x86-64", "x86-64-v3": "x86-64-v3",
         "x86-64-v4": "x86-64-v4"}
CASE_COUNT = 24

V1_FORBIDDEN_EXACT = {
    "addsubpd", "addsubps", "blendpd", "blendps", "blendvpd", "blendvps",
    "adcx", "adox", "aesdec", "aesdeclast", "aesenc", "aesenclast",
    "aesimc", "aeskeygenassist", "andn", "bextr", "blsi", "blsmsk",
    "blsr", "bndcl", "bndcn", "bndcu", "bndldx", "bndmk", "bndmov",
    "bndstx", "bzhi", "cmpxchg16b", "crc32", "dppd", "dpps", "extrq",
    "extractps", "haddpd", "haddps", "hsubpd", "hsubps", "insertps",
    "insertq", "lahf", "lddqu", "lzcnt", "monitor", "movbe", "movddup",
    "movntdqa", "movntsd", "movntss", "movshdup", "movsldup", "mpsadbw",
    "mwait", "pabsb", "pabsd", "pabsw", "packusdw", "palignr", "pblendvb",
    "pblendw", "pcmpestri", "pcmpestrm", "pcmpgtq", "pcmpistri", "pcmpistrm",
    "pcmpeqq", "pclmulqdq", "pextrb", "pextrd", "pextrq", "pextrw",
    "phaddd", "phaddsw", "phaddw", "phminposuw", "phsubd", "phsubsw", "phsubw",
    "pinsrb", "pinsrd", "pinsrq", "pinsrw", "pmaddubsw", "pmuldq", "pmulhrsw",
    "pmulld", "popcnt", "ptest", "pshufb", "rdpid", "rdseed", "rdrand",
    "rorx", "roundpd", "roundps", "roundsd", "roundss", "sahf", "sarx",
    "shlx", "shrx", "tzcnt", "umonitor", "umwait", "wrssd", "wrssq",
    "xsetbv", "xsave", "xsavec", "xsaveopt", "xsaves", "xrstor",
    "xrstors", "xrstors64", "xrstor64",
    "pdep", "pext", "mulx",
    "pmovsxbw", "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq",
    "pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq", "pmovzxdq",
    "pmaxsb", "pmaxsd", "pmaxud", "pmaxuw", "pminsb", "pminsd", "pminud",
    "pminuw",
}
GUARDED_CPU_PROBES = {"xgetbv"}
V1_FORBIDDEN_PREFIXES = ("vfm", "vfma", "vfnm", "vadd", "vsub", "vmul",
                         "vdiv", "vperm", "vblend", "vextract", "vinsert",
                         "vshuf", "vbroadcast", "vzero", "vld", "vst")
def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def capture(command: list[str], *, cwd: Path | None = None,
            env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)


def compiler_metadata(compiler: str) -> dict:
    output = capture([compiler, "--version"])
    if output.returncode:
        raise RuntimeError(f"cannot query compiler {compiler}: {output.stdout}")
    return {"path": compiler, "version": output.stdout.splitlines()[0],
            "full_version_output": output.stdout.strip()}


def dependency_versions() -> dict:
    names = ("numpy", "scipy", "pyFFTW", "matplotlib", "nanobind",
             "pybind11", "scikit-build-core", "cmake", "ninja")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def clean_build_env(compiler: str, profile_lto: bool, helper_path: Path) -> dict:
    env = os.environ.copy()
    for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "DUCC0_CFLAGS",
                "DUCC0_LFLAGS", "DUCC0_FLAGS", "DUCC0_ENABLE_LTO",
                "DUCC0_MAX_PSABI_LEVEL"):
        env.pop(key, None)
    env["CXX"] = compiler
    env["DUCC0_OPTIMIZATION"] = "multiarch"
    env["DUCC0_USE_NANOBIND"] = "1"
    env["DUCC0_BENCH_CMAKE_HELPER"] = str(helper_path)
    env["DUCC0_PROFILE_LOCAL_LTO"] = "ON" if profile_lto else "OFF"
    return env


def run_logged(command: list[str], log_path: Path, label: str, cwd: Path,
               env: dict, timeout: int | None = None) -> tuple[int, float, str]:
    start = time.perf_counter()
    result = capture(command, cwd=cwd, env=env, timeout=timeout)
    duration = time.perf_counter() - start
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {label}; elapsed={duration:.3f}s; exit={result.returncode} =====\n")
        log.write("COMMAND: " + shlex.join(command) + "\n")
        log.write(result.stdout)
        if not result.stdout.endswith("\n"):
            log.write("\n")
    return result.returncode, duration, result.stdout


def configure_and_build(config: dict, source: Path, artifact: Path,
                        build_root: Path, compiler: str, jobs: int) -> dict:
    variant = config["id"]
    build_dir = build_root / "build" / variant
    install_dir = build_root / "install" / variant
    log_path = artifact / "build-logs" / f"{variant}.log"
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    version = tomllib.loads((source / "pyproject.toml").read_text())[
        "project"]["version"]
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake is not installed")
    env = clean_build_env(compiler, config["profile_lto"], LTO_HELPER)
    configure_cmd = [
        cmake, "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DSKBUILD_PROJECT_NAME=ducc0",
        f"-DSKBUILD_PROJECT_VERSION={version}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DDUCC0_BENCH_CMAKE_HELPER={LTO_HELPER}",
        f"-DDUCC0_PROFILE_LOCAL_LTO={'ON' if config['profile_lto'] else 'OFF'}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    diagnostics = {"source_directory": str(source),
                   "configure_command": shlex.join(configure_cmd),
                   "build_command": None, "install_command": None,
                   "log": str(log_path)}
    configure_rc, configure_s, configure_output = run_logged(
        configure_cmd, log_path, "configure", source, env)
    diagnostics["configure_seconds"] = configure_s
    if configure_rc:
        raise BuildFailure("CMake configure failed", diagnostics, configure_output)

    build_cmd = [cmake, "--build", str(build_dir), "--parallel", str(jobs), "--verbose"]
    diagnostics["build_command"] = shlex.join(build_cmd)
    build_rc, build_s, build_output = run_logged(build_cmd, log_path, "build", source, env)
    diagnostics["build_seconds"] = build_s
    if build_rc:
        raise BuildFailure("CMake build failed", diagnostics, build_output)

    install_cmd = [cmake, "--install", str(build_dir)]
    diagnostics["install_command"] = shlex.join(install_cmd)
    install_rc, install_s, install_output = run_logged(
        install_cmd, log_path, "install", source, env)
    diagnostics["install_seconds"] = install_s
    if install_rc:
        raise BuildFailure("CMake install failed", diagnostics, install_output)

    extensions = sorted(install_dir.glob("ducc0*.so"))
    if len(extensions) != 1:
        raise BuildFailure(
            f"expected one installed DUCC extension, found {extensions}", diagnostics,
            install_output)
    diagnostics.update({
        "build_directory": str(build_dir),
        "install_directory": str(install_dir),
        "extension": str(extensions[0]),
        "extension_size_bytes": extensions[0].stat().st_size,
        "total_build_seconds": diagnostics["configure_seconds"]
            + diagnostics["build_seconds"] + diagnostics["install_seconds"],
        "warning_count": sum(1 for line in (configure_output + build_output + install_output)
                             .splitlines() if "warning:" in line.lower()),
        "warning_lines": [line.strip() for line in
                          (configure_output + build_output + install_output).splitlines()
                          if "warning:" in line.lower()][:30],
    })
    return diagnostics


class BuildFailure(RuntimeError):
    def __init__(self, message: str, diagnostics: dict, tail: str):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.tail = tail[-12000:]


def command_tokens(entry: dict) -> list[str]:
    if "arguments" in entry:
        return entry["arguments"]
    return shlex.split(entry["command"])


def find_compile_commands(build_dir: Path) -> list[dict]:
    path = build_dir / "compile_commands.json"
    if not path.exists():
        raise RuntimeError("compile_commands.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_lto(tokens: list[str]) -> bool:
    return any(token.startswith("-flto") for token in tokens)


def find_final_link_command(log_text: str) -> str:
    lines = [line.strip() for line in log_text.splitlines()
             if ".so" in line and " -o " in line and
             ("g++" in line or "c++" in line)]
    if not lines:
        raise RuntimeError("verbose build log has no final extension linker command")
    return lines[-1]


def validate_commands(config: dict, diagnostics: dict) -> dict:
    build_dir = Path(diagnostics["build_directory"])
    commands = find_compile_commands(build_dir)
    profiles = {}
    for name in ("v1", "v3", "v4"):
        profile_name = {"v1": "ducc0_v1", "v3": "ducc0_v3", "v4": "ducc0_v4"}[name]
        entries = [entry for entry in commands
                   if any(profile_name in token for token in command_tokens(entry))]
        if not entries:
            raise RuntimeError(f"no compile commands found for {profile_name}")
        expected_march = {"v1": "x86-64", "v3": "x86-64-v3", "v4": "x86-64-v4"}[name]
        for entry in entries:
            tokens = command_tokens(entry)
            if f"-march={expected_march}" not in tokens:
                raise RuntimeError(
                    f"{profile_name} compile command lacks -march={expected_march}: {entry}")
            if "-fPIC" not in tokens:
                raise RuntimeError(f"{profile_name} compile command is not PIC: {entry}")
            if _has_lto(tokens) != config["profile_lto"]:
                raise RuntimeError(
                    f"{profile_name} compile LTO state differs from requested factor")
        profiles[name] = {"compile_commands": len(entries), "march": expected_march,
                          "pic": True, "lto": config["profile_lto"]}
    if len({value["compile_commands"] for value in profiles.values()}) != 1:
        raise RuntimeError(f"profile translation-unit counts differ: {profiles}")

    dispatcher_entries = [entry for entry in commands
                          if entry.get("file", "").endswith(("python/ducc_driver.cc",
                                                              "python/multiarch.cc"))]
    if len(dispatcher_entries) != 2:
        raise RuntimeError(f"expected baseline dispatcher sources, found {dispatcher_entries}")
    for entry in dispatcher_entries:
        tokens = command_tokens(entry)
        if "-march=x86-64" not in tokens:
            raise RuntimeError(f"dispatcher is not compiled for baseline x86-64: {entry}")
        if any(token in tokens for token in ("-march=native", "-march=x86-64-v3",
                                              "-march=x86-64-v4")):
            raise RuntimeError(f"dispatcher compile command enables a higher ISA: {entry}")
        if _has_lto(tokens):
            raise RuntimeError(f"dispatcher compile command unexpectedly uses LTO: {entry}")
        if "-fno-lto" not in tokens:
            raise RuntimeError(f"dispatcher lacks an explicit -fno-lto: {entry}")

    log_text = Path(diagnostics["log"]).read_text(encoding="utf-8", errors="replace")
    final_link = find_final_link_command(log_text)
    final_tokens = shlex.split(final_link)
    if _has_lto(final_tokens) or "-fno-lto" not in final_tokens:
        raise RuntimeError(f"final extension link is not explicitly non-LTO: {final_link}")

    partial_commands = [line.strip() for line in log_text.splitlines()
                        if "-flinker-output=nolto-rel" in line]
    if config["profile_lto"]:
        if len(partial_commands) != 3:
            raise RuntimeError(f"expected 3 profile-local partial-link commands, found {len(partial_commands)}")
        for name in ("v1", "v3", "v4"):
            expected_march = {"v1": "x86-64", "v3": "x86-64-v3", "v4": "x86-64-v4"}[name]
            candidates = [line for line in partial_commands
                          if f"ducc0_{name}.native.o" in line]
            if len(candidates) != 1:
                raise RuntimeError(f"no unique {name} native partial-link command")
            tokens = shlex.split(candidates[0])
            if "-flto" not in tokens or "-r" not in tokens or "-fPIC" not in tokens:
                raise RuntimeError(f"incomplete profile-local partial-link flags: {candidates[0]}")
            if ("-flinker-output=nolto-rel" not in tokens or
                    "-flto-partition=none" not in tokens):
                raise RuntimeError(f"partial link does not emit partitioned native code: {candidates[0]}")
            if f"-march={expected_march}" not in tokens:
                raise RuntimeError(f"wrong target architecture in partial link: {candidates[0]}")
            if not any(f"ducc0_lib_{name}.dir" in token for token in tokens):
                raise RuntimeError(f"partial link includes no objects from profile {name}")
            own_objects = [token for token in tokens
                           if f"ducc0_lib_{name}.dir" in token and token.endswith(".o")]
            if len(own_objects) != profiles[name]["compile_commands"]:
                raise RuntimeError(
                    f"{name} partial link has {len(own_objects)} profile objects, "
                    f"expected {profiles[name]['compile_commands']}")
            if any(f"ducc0_lib_{other}.dir" in token for token in tokens
                   for other in ("v1", "v3", "v4") if other != name):
                raise RuntimeError(f"cross-profile object contamination in {name} partial link")
        for name in ("v1", "v3", "v4"):
            if f"ducc0_{name}.native.o" not in final_link:
                raise RuntimeError(f"final non-LTO link omits {name} native profile object")
        if any(f"ducc0_lib_{name}.dir" in final_link for name in ("v1", "v3", "v4")):
            raise RuntimeError("final non-LTO link directly consumes profile LTO objects")
    else:
        if partial_commands:
            raise RuntimeError("LTO-OFF build unexpectedly performed an LTO partial link")
        for name in ("v1", "v3", "v4"):
            if f"ducc0_lib_{name}.dir" not in final_link:
                raise RuntimeError(f"final no-LTO link omits direct {name} profile objects")

    diagnostics["profile_compile_validation"] = profiles
    diagnostics["dispatcher_compile_validation"] = {
        "sources": [entry["file"] for entry in dispatcher_entries],
        "march": "x86-64", "lto": False,
    }
    diagnostics["partial_link_commands"] = partial_commands
    diagnostics["final_link_command"] = final_link
    diagnostics["final_link_lto"] = False
    diagnostics["lto_implementation"] = (
        "three independent GCC -r -flto -flto-partition=none "
        "-flinker-output=nolto-rel links, "
        "one object-library profile each" if config["profile_lto"] else
        "direct native profile-object link; LTO disabled")
    return {"commands_valid": True, "profile_commands": profiles,
            "dispatcher_commands": diagnostics["dispatcher_compile_validation"],
            "partial_links": partial_commands, "final_link": final_link}


def _decode_disassembly(object_path: Path) -> dict:
    result = capture(["objdump", "-d", "-M", "intel", str(object_path)])
    if result.returncode:
        raise RuntimeError(f"objdump failed for {object_path}: {result.stdout[-3000:]}")
    count = 0
    vex = 0
    evex = 0
    forbidden = []
    cpu_probes = []
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        tail = line.split(":", 1)[1].strip()
        fields = tail.split()
        byte_fields = []
        for field in fields:
            if re.fullmatch(r"[0-9a-fA-F]{2}", field):
                byte_fields.append(field)
            else:
                break
        if not byte_fields or len(fields) <= len(byte_fields):
            continue
        first = int(byte_fields[0], 16)
        mnemonic = fields[len(byte_fields)].lower()
        count += 1
        if first in (0xC4, 0xC5):
            vex += 1
        elif first == 0x62:
            evex += 1
        if mnemonic in GUARDED_CPU_PROBES:
            cpu_probes.append({"mnemonic": mnemonic, "line": line.strip()})
            continue
        if first in (0xC4, 0xC5, 0x62):
            forbidden.append({"mnemonic": mnemonic, "encoding": f"0x{first:02x}",
                              "line": line.strip()})
        elif mnemonic in V1_FORBIDDEN_EXACT or mnemonic.startswith(V1_FORBIDDEN_PREFIXES):
            forbidden.append({"mnemonic": mnemonic, "line": line.strip()})
        elif re.fullmatch(r"v[a-z0-9_.]+", mnemonic):
            forbidden.append({"mnemonic": mnemonic, "line": line.strip()})
    return {"object": str(object_path), "instruction_count": count,
            "vex_instruction_count": vex, "evex_instruction_count": evex,
            "v1_incompatible_instructions": forbidden[:40],
            "v1_incompatible_count": len(forbidden),
            "guarded_cpu_probe_instructions": cpu_probes,
            "guarded_cpu_probe_count": len(cpu_probes)}


def _symbols(object_path: Path) -> str:
    result = capture(["nm", "-C", str(object_path)])
    if result.returncode:
        raise RuntimeError(f"nm failed for {object_path}: {result.stdout[-3000:]}")
    return result.stdout


def validate_native_object(path: Path, profile: str) -> dict:
    file_result = capture(["file", str(path)])
    header = capture(["readelf", "-h", str(path)])
    sections = capture(["readelf", "-S", str(path)])
    symbols = _symbols(path)
    if file_result.returncode or header.returncode or sections.returncode:
        raise RuntimeError(f"cannot inspect native profile output {path}")
    if "relocatable" not in file_result.stdout.lower() or not re.search(
            r"Type:\s+REL\b", header.stdout):
        raise RuntimeError(f"profile output is not a relocatable ELF object: {path}")
    if ".gnu.lto" in sections.stdout:
        raise RuntimeError(f"LTO IR remains in native profile output: {path}")
    if f"ducc0_{profile}::" not in symbols:
        raise RuntimeError(f"native output has no expected {profile} DUCC symbols: {path}")
    return {"path": str(path), "file": file_result.stdout.strip(),
            "elf_type": "REL", "gnu_lto_sections": False,
            "profile_symbols": sum(1 for line in symbols.splitlines()
                                    if f"ducc0_{profile}::" in line),
            "size_bytes": path.stat().st_size}


def validate_isa(config: dict, diagnostics: dict) -> dict:
    build_dir = Path(diagnostics["build_directory"])
    variant = config["id"]
    if config["profile_lto"]:
        profile_objects = {
            profile: build_dir / f"ducc0_{profile}.native.o"
            for profile in ("v1", "v3", "v4")
        }
    else:
        profile_objects = {}
        for profile in ("v1", "v3", "v4"):
            candidates = list((build_dir / "CMakeFiles" /
                               f"ducc0_lib_{profile}.dir").rglob("fft_inst1.cc.o"))
            if len(candidates) != 1:
                raise RuntimeError(f"cannot find representative {profile} FFT object")
            profile_objects[profile] = candidates[0]

    native_notes = {}
    for profile, object_path in profile_objects.items():
        if not object_path.exists():
            raise RuntimeError(f"missing {profile} profile object: {object_path}")
        if config["profile_lto"]:
            native_notes[profile] = validate_native_object(object_path, profile)
        disassembly = _decode_disassembly(object_path)
        if profile == "v1" and disassembly["v1_incompatible_count"]:
            raise RuntimeError(f"v1 disassembly has higher-ISA instructions: {disassembly}")
        if profile == "v3" and disassembly["evex_instruction_count"]:
            raise RuntimeError(f"v3 disassembly contains EVEX instructions: {disassembly}")
        if disassembly["guarded_cpu_probe_count"]:
            raise RuntimeError(f"profile code contains a CPU-detection probe: {disassembly}")
        disassembly["allowed_profile"] = profile
        disassembly["status"] = "pass"
        native_notes.setdefault(profile, {})["disassembly"] = disassembly

    dispatcher_objects = []
    dispatcher_names = []
    for relative in ("ducc_driver.cc.o", "multiarch.cc.o"):
        candidates = list((build_dir / "CMakeFiles/ducc0.dir").rglob(relative))
        if len(candidates) != 1:
            raise RuntimeError(f"cannot find unique baseline dispatcher object {relative}")
        dispatcher_objects.append(candidates[0])
        dispatcher_names.append(relative)
    dispatcher_notes = [_decode_disassembly(path) for path in dispatcher_objects]
    unsafe_dispatcher = [note for note in dispatcher_notes
                         if note["v1_incompatible_count"] or note["evex_instruction_count"]]
    if unsafe_dispatcher:
        raise RuntimeError(f"dispatcher contains higher-ISA instructions: {unsafe_dispatcher}")
    for name, note in zip(dispatcher_names, dispatcher_notes):
        expected_probes = 1 if name == "multiarch.cc.o" else 0
        if note["guarded_cpu_probe_count"] != expected_probes:
            raise RuntimeError(
                f"unexpected guarded CPU probes in {name}: {note['guarded_cpu_probe_instructions']}")
    source_text = (Path(diagnostics["source_directory"]) / "python/multiarch.cc").read_text(
        encoding="utf-8")
    xgetbv_guard = "if (features.xsave && features.osxsave) xcr0 = read_xcr(0);"
    cpuid_guards = (
        "features.xsave = has_bit(leaf1.ecx, 26);",
        "features.osxsave = has_bit(leaf1.ecx, 27);",
        xgetbv_guard,
    )
    if any(source_text.count(guard) != 1 for guard in cpuid_guards):
        raise RuntimeError("upstream xgetbv CPU feature guard differs from the reviewed source")

    extension = Path(diagnostics["extension"])
    sections = capture(["readelf", "-S", str(extension)])
    if sections.returncode or ".gnu.lto" in sections.stdout:
        raise RuntimeError("final Python extension contains LTO IR or could not be inspected")
    exported = capture(["nm", "-D", "--defined-only", str(extension)])
    if exported.returncode or "PyInit_ducc0" not in exported.stdout:
        raise RuntimeError("final Python extension lost its Python initialization symbol")

    note = {
        "variant": variant,
        "status": "pass",
        "compiled_profiles": list(profile_objects),
        "profile_disassembly": native_notes,
        "dispatcher_disassembly": [
            {**note, "status": "pass", "allowed_profile": "x86-64"}
            for note in dispatcher_notes],
        "dispatcher_objects": [str(path) for path in dispatcher_objects],
        "guarded_cpu_feature_probes": [{
            "instruction": "xgetbv", "count": 1,
            "object": str(dispatcher_objects[1]),
            "guard": "CPUID XSAVE and OSXSAVE feature bits are both checked before read_xcr(0)",
            "source_guard_validated": True,
        }],
        "extension_gnu_lto_sections": False,
        "python_init_symbol": True,
        "lto_state": "profile-local, native partial objects" if config["profile_lto"]
                     else "disabled",
    }
    diagnostics["isa_validation"] = note
    return note


def import_variant(config: dict, diagnostics: dict, artifact: Path,
                   python_executable: str) -> dict:
    install_dir = Path(diagnostics["install_directory"])
    env = os.environ.copy()
    env.pop("DUCC0_MAX_PSABI_LEVEL", None)
    env["PYTHONPATH"] = str(install_dir) + os.pathsep + env.get("PYTHONPATH", "")
    output = capture([python_executable, "-c",
                      "import json,ducc0; print(json.dumps(ducc0.misc.cpu_info()))"], env=env)
    if output.returncode:
        raise RuntimeError(f"multiarch import failed for {config['id']}: {output.stdout[-4000:]}")
    try:
        info = json.loads(output.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"cannot parse cpu_info for {config['id']}: {output.stdout}") from exc
    if info.get("compiled_profiles") != ["x86-64", "x86-64-v3", "x86-64-v4"]:
        raise RuntimeError(f"{config['id']} compiled profile set is incomplete: {info}")
    if not info.get("available_profiles") or info["available_profiles"][0] != "x86-64":
        raise RuntimeError(f"{config['id']} has no runnable v1 profile: {info}")
    if info.get("active_profile") != info["available_profiles"][-1]:
        raise RuntimeError(f"{config['id']} does not select the best available profile: {info}")
    return info


def _defined_demangled_symbols(extension: Path) -> list[tuple[str, str]]:
    result = capture(["nm", "-C", "--defined-only", str(extension)])
    if result.returncode:
        raise RuntimeError(f"nm failed for {extension}: {result.stdout[-3000:]}")
    symbols = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*[0-9a-fA-F]+\s+([A-Za-z])\s+(.+)$", line)
        if match:
            symbols.append((match.group(1), match.group(2)))
    return symbols


def _disassemble_symbol(extension: Path, symbol: str) -> tuple[str, list[tuple[str, str]]]:
    result = capture(["objdump", "-d", "-C", "-M", "intel",
                      f"--disassemble={symbol}", str(extension)])
    if result.returncode:
        raise RuntimeError(f"objdump failed for {symbol}: {result.stdout[-3000:]}")
    instructions = []
    for line in result.stdout.splitlines():
        address_match = re.match(r"^\s*([0-9a-fA-F]+):\s*(.*)$", line)
        if not address_match:
            continue
        fields = address_match.group(2).split()
        byte_count = 0
        while byte_count < len(fields) and re.fullmatch(r"[0-9a-fA-F]{2}", fields[byte_count]):
            byte_count += 1
        if byte_count == 0 or byte_count >= len(fields):
            continue
        address = int(address_match.group(1), 16)
        instructions.append((f"{address:x}",
                             " ".join(fields[byte_count:]).strip()))
    if not instructions:
        raise RuntimeError(f"objdump found no instructions for {symbol} in {extension}")
    return result.stdout, instructions


def _canonical_instructions(instructions: list[tuple[str, str]]) -> list[str]:
    start = int(instructions[0][0], 16)
    canonical = []
    for address_text, body in instructions:
        fields = body.split(None, 1)
        mnemonic = fields[0].lower()
        operands = fields[1].lower() if len(fields) == 2 else ""
        if (mnemonic == "call" or mnemonic.startswith("j") or
                mnemonic.startswith("loop") or mnemonic == "xbegin"):
            target = re.match(r"^(?:0x)?([0-9a-f]+)(?:\s+<([^>]+)>)?", operands)
            if target:
                annotation = target.group(2)
                if annotation:
                    operands = f"target<{annotation}>" + operands[target.end():]
                else:
                    relative = int(target.group(1), 16) - start
                    operands = f"target+{relative:x}" + operands[target.end():]
        operands = re.sub(r"\[rip[+-]0x[0-9a-f]+\]", "[rip+reloc]", operands)
        if "[rip+reloc]" in operands:
            operands = re.sub(r"\s*#.*$", "", operands)
        else:
            operands = re.sub(r"#\s*(?:0x)?[0-9a-f]+\s*(<[^>]+>)?",
                              lambda match: "#" + (match.group(1) or "reloc"), operands)
        canonical.append(f"{mnemonic} {operands}".rstrip())
    return canonical


def _rip_reference_targets(instructions: list[tuple[str, str]]) -> list[str]:
    targets = []
    for _, body in instructions:
        if "[rip" not in body.lower():
            continue
        comment = body.partition("#")[2].strip()
        targets.append(comment or "unresolved target")
    return targets


def _reference_identity(target: str) -> tuple[str, str]:
    match = re.search(r"<(.+)>$", target)
    symbol = match.group(1) if match else target
    symbol = re.sub(r"\+0x[0-9a-fA-F]+$", "+reloc", symbol)
    base_symbol = re.sub(r"\+reloc$", "", symbol)
    return symbol, base_symbol


def _run_reverse_correctness(extension_dir: Path, profile: str,
                             python_executable: str, build_log: Path) -> dict:
    script_dir = str(SCRIPT_DIR)
    script = r'''import json
import ducc0
import numpy as np
from fft_factorial_benchmark import (CASES, _reference_transform, _result_dtype,
    call_ducc, make_input, output_shape, relative_l2_error, tolerance_for)

profile = __import__("os").environ["DUCC0_MAX_PSABI_LEVEL"]
expected = {"1": "x86-64", "3": "x86-64-v3"}[profile]
info = ducc0.misc.cpu_info()
assert info["active_profile"] == expected, info
assert info["configured_limit"] == expected, info
assert info["compiled_profiles"] == ["x86-64", "x86-64-v3", "x86-64-v4"], info
print(json.dumps({"record_type": "import", "profile": expected,
                  "cpu_info": info}), flush=True)
rows = []
for case in CASES:
    print(json.dumps({"record_type": "case_start", "case": case["id"]}), flush=True)
    shape, data = make_input(case, 0)
    reference = _reference_transform("numpy", case["operation"], data, shape, 1)
    out = np.empty(output_shape(case["operation"], shape), dtype=_result_dtype(case))
    result = call_ducc(ducc0, case["operation"], data, shape, out)
    error = relative_l2_error(result, reference)
    passed = (tuple(result.shape) == output_shape(case["operation"], shape)
              and np.isfinite(error) and error <= tolerance_for(case))
    rows.append({"case": case["id"], "shape": list(shape), "l2_error": error,
                 "tolerance": tolerance_for(case), "correctness": "pass" if passed else "fail"})
    print(json.dumps({"record_type": "case_result", **rows[-1]}, sort_keys=True), flush=True)
    if not passed:
        break
print(json.dumps({"record_type": "summary", "profile": expected,
                  "cpu_info": info, "case_count": len(rows),
                  "correctness": "pass" if len(rows) == len(CASES) and
                      all(row["correctness"] == "pass" for row in rows) else "fail",
                  "cases": rows}, sort_keys=True), flush=True)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(extension_dir) + os.pathsep + script_dir +
                         os.pathsep + env.get("PYTHONPATH", ""))
    env["DUCC0_MAX_PSABI_LEVEL"] = profile
    result = capture([python_executable, "-c", script], cwd=extension_dir,
                     env=env, timeout=1800)
    with build_log.open("a", encoding="utf-8") as log:
        log.write(f"\n===== reversed extension correctness at {profile}; exit={result.returncode} =====\n")
        log.write(result.stdout)
        if not result.stdout.endswith("\n"):
            log.write("\n")
    records = []
    for line in result.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    ready = next((row for row in records if row.get("record_type") == "import"), None)
    summary = next((row for row in records if row.get("record_type") == "summary"), None)
    completed = [row for row in records if row.get("record_type") == "case_result"]
    started = [row for row in records if row.get("record_type") == "case_start"]
    if ready is None:
        return {"profile": {"1": "x86-64", "3": "x86-64-v3"}[profile],
                "import": "fail", "correctness": "not-run",
                "process_returncode": result.returncode,
                "termination_signal": {-4: "SIGILL", -7: "SIGBUS", -11: "SIGSEGV"}.get(
                    result.returncode),
                "error": result.stdout[-4000:] or "reversed extension import failed"}
    if summary and result.returncode == 0:
        details = summary
        details["import"] = "pass"
        details["process_returncode"] = 0
        details["termination_signal"] = None
        return details
    active_case = started[-1]["case"] if started else None
    return {
        "profile": {"1": "x86-64", "3": "x86-64-v3"}[profile],
        "cpu_info": ready["cpu_info"],
        "import": "pass",
        "correctness": "fail",
        "case_count": len(completed),
        "completed_cases": completed,
        "failed_or_interrupted_case": active_case,
        "process_returncode": result.returncode,
        "termination_signal": {-4: "SIGILL", -7: "SIGBUS", -11: "SIGSEGV"}.get(
            result.returncode),
        "error": result.stdout[-4000:] or f"worker exited with status {result.returncode}",
    }


def validate_reverse_final_link(config: dict, diagnostics: dict, common_info: dict,
                                artifact: Path, python_executable: str) -> dict:
    if config["id"] != "E" or not config["profile_lto"]:
        raise RuntimeError("reverse final-link check is restricted to LTO configuration E")
    if not diagnostics.get("command_validation", {}).get("commands_valid"):
        raise RuntimeError("E profile-local partial links have not passed command validation")
    if diagnostics.get("isa_validation", {}).get("status") != "pass":
        raise RuntimeError("E native profile objects have not passed their ISA audit")
    partial_commands = list(diagnostics.get("partial_link_commands", []))
    if len(partial_commands) != 3:
        raise RuntimeError(f"E must have exactly three audited profile-local partial links, got {len(partial_commands)}")
    partial_links_sha256 = sha256_bytes("\n".join(partial_commands).encode())

    build_dir = Path(diagnostics["build_directory"])
    normal_extension = Path(diagnostics["extension"])
    original = shlex.split(diagnostics["final_link_command"])
    compiler_index = next((index for index, token in enumerate(original)
                           if any(name in Path(token).name
                                  for name in ("c++", "g++", "clang++"))), None)
    if compiler_index is None:
        raise RuntimeError("cannot locate the C++ linker in E's recorded final link")
    tokens = original[compiler_index:]
    if "&&" in tokens:
        tokens = tokens[:tokens.index("&&")]
    normal_tokens = list(tokens)
    object_positions = {}
    for profile in ("v1", "v3", "v4"):
        matches = [index for index, token in enumerate(tokens)
                   if Path(token).name == f"ducc0_{profile}.native.o"]
        if len(matches) != 1:
            raise RuntimeError(f"E final link does not contain one {profile} native object")
        object_positions[profile] = matches[0]
    normal_order = [profile for _, profile in sorted(
        (position, profile) for profile, position in object_positions.items())]
    if normal_order != ["v1", "v3", "v4"]:
        raise RuntimeError(f"normal E final link order changed unexpectedly: {normal_order}")
    profile_args = {profile: tokens[position]
                    for profile, position in object_positions.items()}
    for position, profile in zip(sorted(object_positions.values()), ("v4", "v3", "v1")):
        tokens[position] = profile_args[profile]

    reverse_dir = build_dir / "reverse-order"
    reverse_dir.mkdir(parents=True, exist_ok=True)
    reverse_extension = reverse_dir / normal_extension.name
    if "-o" not in tokens:
        raise RuntimeError("E final link command has no output argument")
    output_index = tokens.index("-o")
    tokens[output_index + 1] = str(reverse_extension)
    dependency_flags = [index for index, token in enumerate(tokens)
                        if token.startswith("-Wl,--dependency-file=")]
    if len(dependency_flags) != 1:
        raise RuntimeError("E final link has no unique Ninja dependency-file option")
    tokens[dependency_flags[0]] = "-Wl,--dependency-file=CMakeFiles/ducc0.dir/link.reverse-order.d"
    if "-fno-lto" not in tokens or _has_lto(tokens):
        raise RuntimeError("reverse final link must use -fno-lto and contain no LTO flags")
    reverse_order = [profile for _, profile in sorted(
        (tokens.index(profile_args[profile]), profile) for profile in profile_args)]
    if reverse_order != ["v4", "v3", "v1"]:
        raise RuntimeError(f"reverse final-link profile order is wrong: {reverse_order}")
    def other_link_inputs(command_tokens: list[str]) -> list[str]:
        return [token for token in command_tokens
                if token.endswith((".o", ".a")) and
                not any(Path(token).name == f"ducc0_{profile}.native.o"
                        for profile in ("v1", "v3", "v4"))]

    normal_other_inputs = other_link_inputs(normal_tokens)
    reversed_other_inputs = other_link_inputs(tokens)
    if normal_other_inputs != reversed_other_inputs:
        raise RuntimeError("reverse link changed baseline dispatcher or other link inputs")

    link_started = time.perf_counter()
    link_env = clean_build_env(tokens[0], True, LTO_HELPER)
    link = capture(tokens, cwd=build_dir, env=link_env, timeout=900)
    link_seconds = time.perf_counter() - link_started
    command_text = shlex.join(tokens)
    build_log = Path(diagnostics["log"])
    with build_log.open("a", encoding="utf-8") as log:
        log.write("\n===== supplemental E reverse-order final link =====\n")
        log.write(f"profile object order: v4, v3, v1; elapsed={link_seconds:.3f}s\n")
        log.write("COMMAND: " + command_text + "\n")
        log.write(link.stdout)
        if not link.stdout.endswith("\n"):
            log.write("\n")
    if link.returncode or not reverse_extension.exists():
        raise RuntimeError(f"E reverse-order final link failed: {link.stdout[-5000:]}")
    logged_partial_commands = [line.strip() for line in build_log.read_text(
        encoding="utf-8", errors="replace").splitlines()
        if "-flinker-output=nolto-rel" in line]
    if logged_partial_commands != partial_commands:
        raise RuntimeError("reverse final link changed E's three profile-local partial-link commands")

    sections = capture(["readelf", "-S", str(reverse_extension)])
    if sections.returncode or ".gnu.lto" in sections.stdout:
        raise RuntimeError("E reversed extension contains .gnu.lto sections or readelf failed")
    exported = capture(["nm", "-D", "--defined-only", str(reverse_extension)])
    if exported.returncode or not re.search(r"\bPyInit_ducc0\b", exported.stdout):
        raise RuntimeError("E reversed extension does not preserve PyInit_ducc0")
    defined = _defined_demangled_symbols(reverse_extension)
    profile_symbol_counts = {
        profile: sum(1 for _, name in defined if f"ducc0_{profile}::" in name)
        for profile in ("v1", "v3", "v4")
    }
    if any(count == 0 for count in profile_symbol_counts.values()):
        raise RuntimeError(f"E reversed extension lost expected profile symbols: {profile_symbol_counts}")

    normal_defined = _defined_demangled_symbols(normal_extension)
    comparison_symbols = {}
    chosen_profile_symbols = {}
    for profile in ("v1", "v3", "v4"):
        matches = sorted({name for kind, name in normal_defined
                          if kind in "TtWw" and f"ducc0_{profile}::detail_fft::" in name
                          and "::general_r2c<float>(" in name and "[clone .cold]" not in name})
        if not matches:
            raise RuntimeError(f"normal E extension has no general_r2c profile symbol for {profile}")
        chosen_profile_symbols[profile] = matches[0]
    selected = {**{f"{profile}_fft": chosen_profile_symbols[profile]
                   for profile in ("v1", "v3", "v4")},
                "dispatcher_detection": "ducc0_multiarch::detect_cpu_capabilities()",
                "python_initializer": "PyInit_ducc0",
                "shared_std_vector_copy": (
                    "std::vector<unsigned long, std::allocator<unsigned long> >::vector("
                    "std::vector<unsigned long, std::allocator<unsigned long> > const&)")}
    for label, symbol in selected.items():
        if not any(name == symbol for _, name in defined):
            raise RuntimeError(f"reversed extension lost disassembly target {symbol}")
        normal_text, normal_instructions = _disassemble_symbol(normal_extension, symbol)
        reverse_text, reverse_instructions = _disassemble_symbol(reverse_extension, symbol)
        normal_canonical = _canonical_instructions(normal_instructions)
        reverse_canonical = _canonical_instructions(reverse_instructions)
        instruction_changes = [
            {"instruction_index": index,
             "normal": normal_canonical[index] if index < len(normal_canonical) else None,
             "reversed": reverse_canonical[index] if index < len(reverse_canonical) else None}
            for index in range(max(len(normal_canonical), len(reverse_canonical)))
            if (normal_canonical[index] if index < len(normal_canonical) else None) !=
               (reverse_canonical[index] if index < len(reverse_canonical) else None)
        ]
        normal_bytes = "\n".join(body for _, body in normal_instructions)
        reverse_bytes = "\n".join(body for _, body in reverse_instructions)
        normal_refs = _rip_reference_targets(normal_instructions)
        reverse_refs = _rip_reference_targets(reverse_instructions)
        normal_ref_ids = [_reference_identity(item) for item in normal_refs]
        reverse_ref_ids = [_reference_identity(item) for item in reverse_refs]
        changed_references = [
            {"instruction_index": index, "normal_target": normal_refs[index]
             if index < len(normal_refs) else None,
             "reversed_target": reverse_refs[index]
             if index < len(reverse_refs) else None}
            for index in range(max(len(normal_ref_ids), len(reverse_ref_ids)))
            if (normal_ref_ids[index][0] if index < len(normal_ref_ids) else None) !=
               (reverse_ref_ids[index][0] if index < len(reverse_ref_ids) else None)
        ]
        changed_symbols = [item for item in changed_references
                           if _reference_identity(item["normal_target"] or "")[1] !=
                              _reference_identity(item["reversed_target"] or "")[1]]
        normal_encoding = _instruction_bytes(normal_text)
        reverse_encoding = _instruction_bytes(reverse_text)
        vex_count = lambda encoded: sum(
            1 for instruction in encoded
            if instruction.split() and int(instruction.split()[0], 16) in (0xC4, 0xC5))
        evex_count = lambda encoded: sum(
            1 for instruction in encoded
            if instruction.split() and int(instruction.split()[0], 16) == 0x62)
        comparison_symbols[label] = {
            "symbol": symbol,
            "instruction_count": len(normal_instructions),
            "normalized_instruction_stream_match": not instruction_changes,
            "normalized_instruction_change_count": len(instruction_changes),
            "normalized_instruction_change_examples": instruction_changes[:12],
            "raw_instruction_encoding_match": normal_encoding == reverse_encoding,
            "normal_vex_instruction_count": vex_count(normal_encoding),
            "reversed_vex_instruction_count": vex_count(reverse_encoding),
            "normal_evex_instruction_count": evex_count(normal_encoding),
            "reversed_evex_instruction_count": evex_count(reverse_encoding),
            "rip_relative_reference_count": len(normal_refs),
            "rip_relative_reference_changes": len(changed_references),
            "rip_relative_symbol_changes": len(changed_symbols),
            "rip_relative_reference_change_examples": changed_references[:12],
            "normal_instruction_sha256": sha256_bytes(normal_bytes.encode()),
            "reversed_instruction_sha256": sha256_bytes(reverse_bytes.encode()),
            "normal_encoding_sha256": sha256_bytes("\n".join(normal_encoding).encode()),
            "reversed_encoding_sha256": sha256_bytes("\n".join(reverse_encoding).encode()),
        }
        with build_log.open("a", encoding="utf-8") as log:
            log.write(f"\n===== link-order disassembly comparison: {label} =====\n")
            log.write(f"symbol: {symbol}\nnormalized-instruction-stream: {len(normal_canonical)}; "
                      f"changes: {len(instruction_changes)}\n")
            log.write(f"raw-encoding-match: {normal_encoding == reverse_encoding}; "
                      f"RIP-relative reference changes: {len(changed_references)}\n")
            for item in changed_references[:12]:
                log.write("RIP reference difference: " + json.dumps(item, sort_keys=True) + "\n")
            log.write("--- normal order ---\n" + normal_text)
            log.write("--- reversed order ---\n" + reverse_text)

    correctness = {}
    for profile in ("1", "3"):
        correctness[profile] = _run_reverse_correctness(
            reverse_dir, profile, python_executable, build_log)
    if set(correctness) != {"1", "3"}:
        raise RuntimeError("reverse extension correctness must run only at v1 and v3")
    imports_passed = all(item.get("import") == "pass"
                         for item in correctness.values())
    import_failures = [item.get("profile", profile)
                       for profile, item in correctness.items()
                       if item.get("import") != "pass"]
    archived_extension = artifact / "reverse-order-extension" / "E" / normal_extension.name
    archived_extension.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reverse_extension, archived_extension)

    result = {
        "status": ("pass" if all(item.get("import") == "pass" and
                                    item.get("correctness") == "pass"
                                    for item in correctness.values()) and
                   all(item["normalized_instruction_stream_match"]
                       for item in comparison_symbols.values()) else "fail"),
        "normal_profile_object_order": normal_order,
        "reversed_profile_object_order": reverse_order,
        "profile_local_partial_links_reused_unchanged": True,
        "partial_link_command_count": len(partial_commands),
        "partial_link_commands_sha256": partial_links_sha256,
        "partial_links_reexecuted": False,
        "other_link_inputs_unchanged": True,
        "unchanged_other_link_inputs": normal_other_inputs,
        "final_link_command": command_text,
        "final_link_seconds": link_seconds,
        "final_link_fno_lto": "-fno-lto" in tokens,
        "final_link_lto_flags": [token for token in tokens if token.startswith("-flto")],
        "extension": str(archived_extension),
        "build_extension": str(reverse_extension),
        "extension_size_bytes": reverse_extension.stat().st_size,
        "gnu_lto_sections": False,
        "python_init_symbol": "PyInit_ducc0",
        "profile_symbol_counts": profile_symbol_counts,
        "import": ("pass at v1 and v3 configured limits" if imports_passed else
                   "fail at " + ", ".join(import_failures)),
        "imports_by_profile": {
            item.get("profile", {"1": "x86-64", "3": "x86-64-v3"}[profile]):
                item.get("import", "unavailable")
            for profile, item in correctness.items()
        },
        "correctness_profiles": ["x86-64", "x86-64-v3"],
        "correctness": correctness,
        "v4_execution": "not run; compiled, linked, and disassembled only",
        "disassembly_comparison": comparison_symbols,
        "detected_link_order_instruction_changes": [
            label for label, item in comparison_symbols.items()
            if not item["normalized_instruction_stream_match"]],
    }
    if result["status"] != "pass":
        failures = []
        for profile, item in correctness.items():
            if item.get("import") != "pass":
                failures.append(f"{item['profile']} reversed extension import failed")
            elif item.get("correctness") != "pass":
                signal = item.get("termination_signal")
                cause = f" ({signal})" if signal else ""
                failures.append(
                    f"{item['profile']} correctness stopped at "
                    f"{item.get('failed_or_interrupted_case', 'unknown case')}{cause}")
        if result["detected_link_order_instruction_changes"]:
            failures.append("link order changed instruction streams in " + ", ".join(
                result["detected_link_order_instruction_changes"]))
        result["error"] = "; ".join(failures)
    diagnostics["reverse_link"] = result
    return result


def _instruction_bytes(disassembly: str) -> list[str]:
    result = []
    for line in disassembly.splitlines():
        match = re.match(r"^\s*[0-9a-fA-F]+:\s*(.*)$", line)
        if not match:
            continue
        fields = match.group(1).split()
        values = []
        for field in fields:
            if not re.fullmatch(r"[0-9a-fA-F]{2}", field):
                break
            values.append(field.lower())
        if values:
            result.append(" ".join(values))
    return result


def prepare_sources(repo_root: Path, worktree_root: Path, manifest_path: Path,
                    variant_status: dict) -> tuple[dict, dict]:
    worktree_root.mkdir(parents=True, exist_ok=True)
    fetched_base = ensure_base_commit(repo_root)
    configs = []
    prepared = {}
    for config in CONFIGS:
        source = worktree_root / config["id"]
        status = variant_status[config["id"]]
        try:
            subprocess.run(["git", "worktree", "add", "--detach", str(source), BASE_SHA],
                           cwd=repo_root, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
            metadata = prepare_variant(source, config, repo_root)
            status["source_preparation"] = "pass"
            status["source_tree"] = str(source)
            status["source_patch_sha256"] = metadata["source_tree_patch_sha256"]
            prepared[config["id"]] = source
            configs.append(metadata)
        except Exception as exc:
            status["source_preparation"] = "fail"
            status["source_error"] = f"{type(exc).__name__}: {exc}"
            configs.append({**config, "source_preparation": "fail",
                            "source_error": status["source_error"],
                            "upstream_base_sha": BASE_SHA})

    tweak_patch = repo_root / ".github/patches/fft_tweaks.patch"
    manifest = {
        "schema_version": 1,
        "base_upstream": "mreineck/ducc",
        "base_branch": "multiarch",
        "base_sha": BASE_SHA,
        "pinned_base_commit_fetched_from_upstream": fetched_base,
        "base_sha_verified_at_branch_creation": True,
        "upstream_head_at_branch_creation": BASE_SHA,
        "fork_multiarch_head_at_branch_creation": "8aabaa0e925812a440f414e1df5cb817c4b70aa5",
        "benchmark_source_head_sha": run_git(repo_root, "rev-parse", "HEAD").strip(),
        "benchmark_source_sha_from_github_event": os.environ.get("FFT_FACTORIAL_SOURCE_SHA"),
        "source_preparation": "eight detached temporary worktrees from one pinned upstream commit",
        "factors": ["profile-local LTO", "special_mul always-inline", "FFT tweaks"],
        "special_mul_factor": {
            "on": "retain the upstream DUCC0_ALWAYS_INLINE annotation",
            "off": "remove the annotation on detail_fft::special_mul only",
            "portable_macro_definition_changed": False,
        },
        "fft_tweaks_provenance": {
            "tip": "b456d7183ac5e667cff1b769bf5b17e33ebd24cb",
            "commits_inspected": [
                "746ee07d02c01afd68f6faf17a9e9511480e971d",
                "449505438c0fbd1e4544b2ae95653d2b1a150d1b",
                "b456d7183ac5e667cff1b769bf5b17e33ebd24cb",
            ],
            "historical_benchmark_commit_excluded": True,
            "source_patch_sha256": TWEAK_PATCH_SHA256,
            "patch_file_sha256": sha256_bytes(tweak_patch.read_bytes()),
            "source_files": ["src/ducc0/fft/fft1d_impl.h", "src/ducc0/fft/fftnd_impl.h"],
            "namespace_handling_preserved": True,
        },
        "build_helper_sha256": sha256_bytes(LTO_HELPER.read_bytes()),
        "configurations": configs,
    }
    write_json(manifest_path, manifest)
    return prepared, manifest


def launch_worker(config: dict, info: dict, diagnostics: dict, profile: str,
                  nrepeat: int, artifact: Path, python_executable: str):
    install_dir = Path(diagnostics["install_directory"])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["DUCC0_MAX_PSABI_LEVEL"] = str(PROFILE_LEVELS[profile])
    log_path = artifact / "worker-logs" / f"{config['id']}-{profile}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    command = [python_executable, str(BENCHMARK_SCRIPT), "--worker",
               "--variant", config["id"], "--profile", profile,
               "--nrepeat", str(nrepeat)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=log, text=True, bufsize=1, env=env)
    first = process.stdout.readline()
    if not first:
        process.wait()
        log.close()
        raise RuntimeError(f"worker {config['id']}/{profile} exited before ready; see {log_path}")
    ready = json.loads(first)
    actual = ready.get("cpu_info", {})
    if (ready.get("record_type") != "worker_ready" or
            actual.get("active_profile") != profile or
            actual.get("configured_limit") != profile or
            actual.get("available_profiles") != info.get("available_profiles")):
        process.kill()
        process.wait()
        log.close()
        raise RuntimeError(f"worker profile assertion failed for {config['id']}/{profile}: {ready}")
    return {"process": process, "log": log, "log_path": str(log_path),
            "ready": ready}


def stop_worker(worker: dict) -> None:
    process = worker["process"]
    try:
        if process.poll() is None:
            process.stdin.write('{"stop":true}\n')
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=15)
    except Exception:
        process.kill()
        process.wait()
    finally:
        if not worker["log"].closed:
            worker["log"].close()


def run_benchmarks(artifact: Path, buildable: dict, common_info: dict,
                   ntry: int, nrepeat: int, expected_dir: Path) -> dict:
    # Imports are deferred so build diagnostics still exist if a Python package is absent.
    from fft_factorial_benchmark import (
        CASES, REFERENCES, _cpu_identity, measure_reference_sample,
    )

    import json as json_module

    timing_path = artifact / "timings.jsonl"
    reference_path = artifact / "references.jsonl"
    timing_path.write_text("", encoding="utf-8")
    reference_path.write_text("", encoding="utf-8")
    profiles = common_info["available_profiles"]
    measured_variant_ids = [config["id"] for config in CONFIGS
                            if config["id"] in buildable]
    expected_dir.mkdir(parents=True, exist_ok=True)
    worker_logs = []
    reference_failures = []
    worker_failures = []
    runner_cpu_identity = _cpu_identity()
    for profile in profiles:
        workers = {}
        for config in CONFIGS:
            variant = config["id"]
            if variant not in buildable:
                continue
            try:
                workers[variant] = launch_worker(
                    config, common_info, buildable[variant], profile, nrepeat,
                    artifact, sys.executable)
                worker_logs.append(workers[variant]["log_path"])
            except Exception as exc:
                worker_failures.append({"variant": variant, "profile": profile,
                                        "error": f"{type(exc).__name__}: {exc}"})

        try:
            for case_index, case in enumerate(CASES):
                for sample_index in range(ntry):
                    expected_path = expected_dir / profile / case["id"] / f"{sample_index}.npy"
                    rows, input_sha = measure_reference_sample(
                        case, sample_index, profile, nrepeat, expected_path)
                    profile_context = dict(common_info)
                    profile_context["configured_limit"] = profile
                    profile_context["active_profile"] = profile
                    for row in rows:
                        row.update({
                            "compiled_profiles": profile_context["compiled_profiles"],
                            "available_profiles": profile_context["available_profiles"],
                            "configured_profile_limit": profile,
                            "active_profile": profile,
                            "cpu_info": profile_context,
                            "cpu_identity": runner_cpu_identity,
                            "profile_metadata_scope": (
                                "matched DUCC worker context; reference backend is not ISA-matched"),
                        })
                    numpy_witness = next((row for row in rows
                                          if row["reference"] == "numpy" and
                                          row["correctness"] == "pass"), None)
                    if not numpy_witness:
                        reference_failures.append({"profile": profile,
                                                   "case": case["id"],
                                                   "sample_index": sample_index,
                                                   "error": "NumPy correctness witness unavailable"})
                    for row in rows:
                        append_jsonl(reference_path, row)
                        if row["correctness"] != "pass":
                            reference_failures.append({
                                "profile": profile, "case": case["id"],
                                "sample_index": sample_index,
                                "reference": row["reference"],
                                "error": row["correctness_error"],
                            })

                    # Rotate variant order by case and sample; all results remain on this runner.
                    rotation = (case_index + sample_index) % len(CONFIGS)
                    order = [CONFIGS[(rotation + offset) % len(CONFIGS)]["id"]
                             for offset in range(len(CONFIGS))]
                    for order_index, variant in enumerate(order):
                        if variant not in workers:
                            continue
                        worker = workers[variant]
                        process = worker["process"]
                        if process.poll() is not None:
                            worker_failures.append({
                                "variant": variant, "profile": profile,
                                "case": case["id"], "sample_index": sample_index,
                                "error": f"worker exited with status {process.returncode}",
                            })
                            stop_worker(worker)
                            del workers[variant]
                            continue
                        command = {
                            "case": case["id"], "sample_index": sample_index,
                            "input_sha256": input_sha,
                            "expected_path": str(expected_path) if numpy_witness else None,
                        }
                        process.stdin.write(json_module.dumps(command) + "\n")
                        process.stdin.flush()
                        line = process.stdout.readline()
                        if not line:
                            worker_failures.append({
                                "variant": variant, "profile": profile,
                                "case": case["id"], "sample_index": sample_index,
                                "error": "worker closed stdout without a result",
                            })
                            stop_worker(worker)
                            del workers[variant]
                            continue
                        try:
                            record = json_module.loads(line)
                        except json_module.JSONDecodeError as exc:
                            worker_failures.append({
                                "variant": variant, "profile": profile,
                                "case": case["id"], "sample_index": sample_index,
                                "error": f"invalid worker JSON: {exc}",
                            })
                            stop_worker(worker)
                            del workers[variant]
                            continue
                        record["case_rotation"] = rotation
                        record["rotation_order_index"] = order_index
                        append_jsonl(timing_path, record)
                        if record.get("record_type") == "ducc_error":
                            worker_failures.append(record)
                        elif record.get("correctness") != "pass":
                            worker_failures.append({
                                "variant": variant, "profile": profile,
                                "case": case["id"], "sample_index": sample_index,
                                "error": record.get("correctness_error") or
                                    "correctness result unavailable",
                                "l2_error": record.get("l2_error"),
                                "correctness": record.get("correctness"),
                            })
                    try:
                        expected_path.unlink(missing_ok=True)
                        expected_path.with_suffix(expected_path.suffix + ".sha256").unlink(missing_ok=True)
                    except OSError:
                        pass
                print(f"completed profile={profile} case={case['id']}", flush=True)
        finally:
            for worker in list(workers.values()):
                stop_worker(worker)

    return {
        "supported_profiles": profiles,
        "skipped_profiles": [name for name in PROFILE_LEVELS if name not in profiles],
        "measured_variants": measured_variant_ids,
        "reference_reuse": "one FFTW/SciPy/NumPy timing set per profile, case, and sample shared across A-H",
        "variant_order": "rotated by case and sample across the eight IDs",
        "expected_shape_policy": "random cases use pinned-base complex good_size; fixed controls bypass it",
        "ntry": ntry,
        "nrepeat": nrepeat,
        "threads": 1,
        "worker_logs": worker_logs,
        "reference_failures": reference_failures,
        "worker_failures": worker_failures,
    }


def inspect_natural_cpu(buildable: dict, artifact: Path) -> tuple[dict, dict]:
    infos = {}
    errors = {}
    for variant, diagnostics in buildable.items():
        config = next(item for item in CONFIGS if item["id"] == variant)
        try:
            info = import_variant(config, diagnostics, artifact, sys.executable)
            infos[variant] = info
            write_json(artifact / "cpu-info" / f"{variant}.json", info)
        except Exception as exc:
            errors[variant] = f"{type(exc).__name__}: {exc}"
    if errors:
        raise RuntimeError(f"one or more built variants failed import/CPU checks: {errors}")
    unique = {json.dumps(info, sort_keys=True) for info in infos.values()}
    if len(unique) != 1:
        raise RuntimeError(f"variant CPU/profile selection differs: {infos}")
    common = next(iter(infos.values()))
    return common, infos


def git_head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD").strip()


def _get_upstream_head() -> str | None:
    try:
        result = capture(["git", "ls-remote", "https://github.com/mreineck/ducc.git",
                          "refs/heads/multiarch"], timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def main() -> int:
    run_started_perf = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path)
    parser.add_argument("--ntry", type=int, default=3)
    parser.add_argument("--nrepeat", type=int, default=5)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    if args.ntry < 1 or args.nrepeat < 1:
        parser.error("ntry and nrepeat must be positive")

    artifact = args.output_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    worktree_root = (args.worktree_root or
                     Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "fft-factorial-source").resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="fft-factorial-build-",
                                       dir=worktree_root.parent))
    jobs = args.jobs or min(os.cpu_count() or 2, 4)
    compiler_name = os.environ.get("CXX", "g++")
    compiler = shutil.which(compiler_name)
    if not compiler:
        compiler = shutil.which("c++")
    if not compiler:
        raise RuntimeError("no C++ compiler is available")
    compiler = str(Path(compiler).resolve())
    compiler_info = compiler_metadata(compiler)
    versions = dependency_versions()
    source_sha = os.environ.get("FFT_FACTORIAL_SOURCE_SHA") or git_head(ROOT)
    shutil.copy2(ROOT / ".github/patches/fft_tweaks.patch",
                 artifact / "fft_tweaks.patch")
    shutil.copy2(LTO_HELPER, artifact / "profile_local_lto.cmake")
    write_json(artifact / "patch-provenance.json", {
        "base_sha": BASE_SHA,
        "fft_tweaks_tip": "b456d7183ac5e667cff1b769bf5b17e33ebd24cb",
        "fft_tweaks_patch_sha256": sha256_bytes(
            (ROOT / ".github/patches/fft_tweaks.patch").read_bytes()),
        "profile_local_lto_helper_sha256": sha256_bytes(LTO_HELPER.read_bytes()),
        "special_mul_off_patch": "single DUCC0_ALWAYS_INLINE annotation removal",
    })

    variants = {config["id"]: {
        **config,
        "source_preparation": "pending",
        "build": "pending",
        "build_isolation": "pending",
        "isa_validation": "pending",
        "runtime_import": "pending",
        "reverse_link_validation": "pending" if config["id"] == "E" else "not-required",
    } for config in CONFIGS}
    run_status = {
        "schema_version": 1,
        "base_upstream_sha": BASE_SHA,
        "benchmark_source_head_sha": source_sha,
        "benchmark_checkout_head_sha": git_head(ROOT),
        "runner_os": os.environ.get("RUNNER_OS", platform.system()),
        "runner_architecture": platform.machine(),
        "runner_cpu_model": platform.processor() or None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ntry": args.ntry,
        "nrepeat": args.nrepeat,
        "threads": 1,
        "expected_cases_per_profile": CASE_COUNT,
        "expected_configuration_count": 8,
        "compiler": compiler_info,
        "dependencies": versions,
        "build_jobs": jobs,
        "upstream_multiarch_head_at_run": _get_upstream_head(),
        "variants": variants,
        "profiles": [],
        "overall_status": "running",
    }
    write_json(artifact / "run-status.json", run_status)

    manifest_path = artifact / "configuration-manifest.json"
    source_preparation_started = time.perf_counter()
    prepared_sources, source_manifest = prepare_sources(
        ROOT, worktree_root, manifest_path, variants)
    run_status["source_preparation_seconds"] = (
        time.perf_counter() - source_preparation_started)
    run_status["source_manifest"] = str(manifest_path)
    buildable = {}

    build_loop_started = time.perf_counter()
    for config in CONFIGS:
        variant = config["id"]
        status = variants[variant]
        source = prepared_sources.get(variant)
        if source is None:
            status["build"] = "skipped-source-preparation-failed"
            write_json(artifact / "run-status.json", run_status)
            continue
        try:
            diagnostics = configure_and_build(
                config, source, artifact, build_root, compiler, jobs)
            status["build"] = "pass"
            status["build_diagnostics"] = diagnostics
            try:
                command_validation = validate_commands(config, diagnostics)
                status["build_isolation"] = "pass"
                status["build_diagnostics"]["command_validation"] = command_validation
                isa = validate_isa(config, diagnostics)
                status["isa_validation"] = "pass"
                status["build_diagnostics"]["isa_validation"] = isa
                status["runtime_import"] = "pending"
                buildable[variant] = diagnostics
            except Exception as exc:
                status["build_isolation"] = "fail"
                status["isa_validation"] = "fail"
                status["validation_error"] = f"{type(exc).__name__}: {exc}"
        except BuildFailure as exc:
            status["build"] = "fail"
            status["build_diagnostics"] = exc.diagnostics
            status["build_error"] = str(exc)
            status["build_tail"] = exc.tail
        except Exception as exc:
            status["build"] = "fail"
            status["build_error"] = f"{type(exc).__name__}: {exc}"
        write_json(artifact / "run-status.json", run_status)

    run_status["build_loop_seconds"] = time.perf_counter() - build_loop_started

    common_info = None
    import_errors = {}
    for variant in list(buildable):
        try:
            config = next(item for item in CONFIGS if item["id"] == variant)
            info = import_variant(config, buildable[variant], artifact, sys.executable)
            variants[variant]["runtime_import"] = "pass"
            variants[variant]["cpu_info"] = info
            if common_info is None:
                common_info = info
            elif info != common_info:
                raise RuntimeError(f"CPU/profile data differs from other variants: {info}")
        except Exception as exc:
            import_errors[variant] = f"{type(exc).__name__}: {exc}"
            variants[variant]["runtime_import"] = "fail"
            variants[variant]["runtime_import_error"] = import_errors[variant]
            del buildable[variant]
    if common_info is None:
        run_status["cpu_detection_error"] = "no built and audited variant could report cpu_info()"
    else:
        run_status["cpu_info"] = common_info
        run_status["supported_profiles"] = common_info["available_profiles"]
        run_status["skipped_profiles"] = [profile for profile in PROFILE_LEVELS
                                           if profile not in common_info["available_profiles"]]
        run_status["profiles"] = [
            {"name": profile,
             "status": "supported" if profile in common_info["available_profiles"] else "skipped",
             "reason": None if profile in common_info["available_profiles"] else
                 "profile is not in upstream cpu_info().available_profiles"}
            for profile in PROFILE_LEVELS
        ]

    reverse_started = time.perf_counter()
    reverse_status = variants["E"]
    if common_info is not None and "E" in buildable:
        try:
            reverse_note = validate_reverse_final_link(
                next(item for item in CONFIGS if item["id"] == "E"),
                buildable["E"], common_info, artifact, sys.executable)
            reverse_status["reverse_link_validation"] = reverse_note["status"]
            if reverse_note["status"] != "pass":
                reverse_status["reverse_link_error"] = reverse_note.get(
                    "error", "reversed extension correctness checks failed")
            reverse_status["build_diagnostics"]["reverse_link"] = reverse_note
        except Exception as exc:
            reverse_error = f"{type(exc).__name__}: {exc}"
            reverse_status["reverse_link_validation"] = "fail"
            reverse_status["reverse_link_error"] = reverse_error
            reverse_status.setdefault("build_diagnostics", {})["reverse_link"] = {
                "status": "fail", "error": reverse_error,
            }
    else:
        reverse_error = "E did not pass the normal build, ISA audit, and import checks"
        reverse_status["reverse_link_validation"] = "fail"
        reverse_status["reverse_link_error"] = reverse_error
        reverse_status.setdefault("build_diagnostics", {})["reverse_link"] = {
            "status": "fail", "error": reverse_error,
        }
    run_status["reverse_link_validation_seconds"] = time.perf_counter() - reverse_started
    write_json(artifact / "run-status.json", run_status)

    benchmark_summary = None
    benchmark_started = time.perf_counter()
    if common_info is not None and buildable:
        try:
            benchmark_summary = run_benchmarks(
                artifact, buildable, common_info, args.ntry, args.nrepeat,
                worktree_root / "expected-results")
            run_status["benchmark"] = benchmark_summary
        except Exception as exc:
            run_status["benchmark_error"] = f"{type(exc).__name__}: {exc}"
    elif common_info is not None:
        run_status["benchmark_error"] = "all builds failed validation or import"
    run_status["benchmark_seconds"] = time.perf_counter() - benchmark_started

    from fft_factorial_benchmark import CASES
    supported = run_status.get("supported_profiles", [])
    expected_cells = len(CASES) * len(supported) * len(CONFIGS)
    ducc_records = []
    timing_path = artifact / "timings.jsonl"
    if timing_path.exists():
        for line in timing_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("record_type") == "ducc":
                    ducc_records.append(record)
    by_cell = {}
    for record in ducc_records:
        key = (record["profile"], record["case"], record["variant"])
        by_cell.setdefault(key, []).append(record)
    completed_cells = 0
    failed_cells = 0
    incomplete_cells = 0
    for profile in supported:
        for case in CASES:
            for config in CONFIGS:
                records = by_cell.get((profile, case["id"], config["id"]), [])
                if len(records) == args.ntry:
                    completed_cells += 1
                    if any(row.get("correctness") != "pass" for row in records):
                        failed_cells += 1
                else:
                    incomplete_cells += 1
    reference_records = []
    reference_path = artifact / "references.jsonl"
    if reference_path.exists():
        reference_records = [json.loads(line) for line in
                            reference_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()]
    reference_failures = [row for row in reference_records
                          if row.get("correctness") != "pass"]
    expected_reference_samples = len(CASES) * len(supported) * args.ntry * 3
    run_status.update({
        "expected_variant_profile_case_cells": expected_cells,
        "completed_variant_profile_case_cells": completed_cells,
        "failed_correctness_cells": failed_cells,
        "incomplete_variant_profile_case_cells": incomplete_cells,
        "expected_ducc_sample_records": expected_cells * args.ntry,
        "completed_ducc_sample_records": len(ducc_records),
        "expected_reference_sample_records": expected_reference_samples,
        "completed_reference_sample_records": len(reference_records),
        "reference_failure_records": len(reference_failures),
        "import_errors": import_errors,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    build_failures = [variant for variant, status in variants.items()
                      if status.get("build") != "pass"]
    isolation_failures = [variant for variant, status in variants.items()
                          if status.get("build_isolation") != "pass" or
                          status.get("isa_validation") != "pass"]
    import_failures = [variant for variant, status in variants.items()
                       if status.get("runtime_import") != "pass"]
    reverse_link_failures = [variant for variant, status in variants.items()
                             if status.get("reverse_link_validation") == "fail"]
    benchmark_failures = (benchmark_summary or {}).get("worker_failures", [])
    run_status["failures"] = {
        "builds": build_failures,
        "build_or_isa_validation": isolation_failures,
        "runtime_import": import_failures,
        "reverse_link": reverse_link_failures,
        "correctness_cells": failed_cells,
        "missing_cells": incomplete_cells,
        "reference_correctness": len(reference_failures),
        "benchmark_worker_errors": benchmark_failures,
        "benchmark_error": run_status.get("benchmark_error"),
    }
    run_status["total_run_seconds"] = time.perf_counter() - run_started_perf
    success = not any((build_failures, isolation_failures, import_failures,
                       reverse_link_failures, failed_cells, incomplete_cells, reference_failures,
                       benchmark_failures, run_status.get("benchmark_error")))
    run_status["overall_status"] = "pass" if success else "fail"
    write_json(artifact / "run-status.json", run_status)

    write_json(artifact / "build-diagnostics.json", {
        variant: status.get("build_diagnostics", {})
        for variant, status in variants.items()
    })
    write_json(artifact / "isa-validation.json", {
        variant: status.get("build_diagnostics", {}).get("isa_validation", {
            "status": status.get("isa_validation", "unavailable"),
            "error": status.get("validation_error"),
        }) for variant, status in variants.items()
    })
    write_json(artifact / "reverse-link-validation.json",
               variants["E"].get("build_diagnostics", {}).get("reverse_link", {
                   "status": variants["E"].get("reverse_link_validation", "unavailable"),
                   "error": variants["E"].get("reverse_link_error"),
               }))

    shutil.rmtree(build_root, ignore_errors=True)
    run_status["temporary_builds_removed"] = True
    write_json(artifact / "run-status.json", run_status)

    for config in CONFIGS:
        source = worktree_root / config["id"]
        subprocess.run(["git", "worktree", "remove", "--force", str(source)],
                       cwd=ROOT, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    shutil.rmtree(worktree_root / "expected-results", ignore_errors=True)
    if success:
        print("FFT factorial benchmark completed all supported configurations.", flush=True)
        return 0
    print("FFT factorial benchmark is incomplete or failed; see run-status.json.",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    raise SystemExit(main())
