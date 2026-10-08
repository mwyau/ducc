# TODO:
# - make multiarch work on Windows x86_64? Do we care enough?
cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{DUCC0_OPTIMIZATION})
    set (OPTFLAGS "$ENV{DUCC0_OPTIMIZATION}")
else()
    set (OPTFLAGS "native-strip")
endif()
set(MULTIARCH OFF)
set(MULTIARCH_ARCH "")
if (OPTFLAGS MATCHES "^multiarch" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(MULTIARCH ON)
        set(MULTIARCH_ARCH x86)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|AARCH64|arm64|ARM64)$")
        set(MULTIARCH ON)
        set(MULTIARCH_ARCH arm64)
    endif()
endif()
set(PKGNAME ${SKBUILD_PROJECT_NAME})
set(PKGVERSION ${SKBUILD_PROJECT_VERSION})

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

set(CXX_SOURCES
    src/ducc0/fft/fft_inst1.cc src/ducc0/fft/fft_inst2.cc
    src/ducc0/nufft/spreadinterp_inst1.cc src/ducc0/nufft/spreadinterp_inst2.cc
    src/ducc0/healpix/healpix_base.cc src/ducc0/healpix/healpix_tables.cc
    src/ducc0/math/gl_integrator.cc src/ducc0/math/pointing.cc
    src/ducc0/math/gridding_kernel.cc src/ducc0/math/geom_utils.cc
    src/ducc0/math/wigner3j.cc src/ducc0/math/space_filling.cc
    src/ducc0/wgridder/wgridder_inst1.cc
    src/ducc0/wgridder/wgridder_inst2.cc src/ducc0/wgridder/wgridder_inst3.cc
    src/ducc0/infra/string_utils.cc
    src/ducc0/infra/threading.cc src/ducc0/infra/mav.cc src/ducc0/sht/sht.cc)
set(PYMOD_SOURCES
    python/nufft_pymod.cc python/fft_pymod.cc python/sht_pymod.cc
    python/misc_pymod.cc python/totalconvolve_pymod.cc python/wgridder_pymod.cc
    python/misc_pymod.cc python/healpix_pymod.cc
    python/pointingprovider_pymod.cc python/ducc.cc)

if (MULTIARCH)
    set (DEFS ${DEFS} DUCC0_MULTIARCH)
    if (MULTIARCH_ARCH STREQUAL "x86")
        # We only build libraries for PSABI levels 1, 3, and 4.
        # Level 2 will most likely not be much faster than level 1,
        # so we save the space and fall back to level 1 if it is detected.
        set(MULTIARCH_PROFILES v1 v3 v4)
        set(DUCC0_PROFILE_FLAGS_v1 -march=x86-64)
        set(DUCC0_PROFILE_FLAGS_v3 -march=x86-64-v3)
        set(DUCC0_PROFILE_FLAGS_v4 -march=x86-64-v4)
        set(DUCC0_BASELINE_FLAGS -march=x86-64)
        set(DUCC0_ISOLATION_FLAGS)
    else()
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_QUIET TRUE)

        # Keep these as SHELL groups so CMake's option de-duplication cannot
        # remove the final per-profile override if a user supplied the same
        # option earlier through DUCC0_CFLAGS or DUCC0_FLAGS.
        set(DUCC0_PROFILE_FLAGS_neon "SHELL:-march=armv8-a")
        set(DUCC0_BASELINE_FLAGS "SHELL:-march=armv8-a")
        set(DUCC0_ISOLATION_FLAGS "SHELL:-fno-lto")
        set(MULTIARCH_PROFILES neon)
        set(DUCC0_ARM_PROFILE_MASK 2)
        set(DEFS ${DEFS} DUCC0_MULTIARCH_ARM64)
        set(DUCC0_ARM_MODULE_DEFS DUCC0_ARM_BUILD_NEON)

        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a")
        check_cxx_source_compiles(
            "#include <arm_neon.h>\n#ifndef __ARM_NEON\n#error AArch64 Advanced SIMD is required\n#endif\nint main() { return 0; }"
            DUCC0_COMPILER_HAS_ARMV8_NEON)
        if (NOT DUCC0_COMPILER_HAS_ARMV8_NEON)
            message(FATAL_ERROR
                "The selected compiler cannot build the required Linux AArch64 NEON baseline (-march=armv8-a).")
        endif()

        set(CMAKE_REQUIRED_FLAGS
            "-march=armv8-a+sve -msve-vector-bits=scalable")
        check_cxx_source_compiles(
            "#include <arm_sve.h>\n#ifndef __ARM_FEATURE_SVE\n#error SVE target is unavailable\n#endif\nint main() { return 0; }"
            DUCC0_COMPILER_HAS_SCALABLE_SVE)
        if (DUCC0_COMPILER_HAS_SCALABLE_SVE)
            list(APPEND MULTIARCH_PROFILES sve)
            set(DUCC0_PROFILE_FLAGS_sve
                "SHELL:-march=armv8-a+sve -msve-vector-bits=scalable")
            math(EXPR DUCC0_ARM_PROFILE_MASK "${DUCC0_ARM_PROFILE_MASK} | 4")
            list(APPEND DUCC0_ARM_MODULE_DEFS DUCC0_ARM_BUILD_SVE)
            message(STATUS "ARM64 multiarch: scalable SVE profile enabled")
        else()
            message(STATUS
                "ARM64 multiarch: SVE profile disabled; compiler rejected -march=armv8-a+sve -msve-vector-bits=scalable")
        endif()

        set(CMAKE_REQUIRED_FLAGS
            "-march=armv8-a+sve2 -msve-vector-bits=scalable")
        check_cxx_source_compiles(
            "#include <arm_sve.h>\n#ifndef __ARM_FEATURE_SVE2\n#error SVE2 target is unavailable\n#endif\nint main() { return 0; }"
            DUCC0_COMPILER_HAS_SCALABLE_SVE2)
        if (DUCC0_COMPILER_HAS_SCALABLE_SVE2)
            list(APPEND MULTIARCH_PROFILES sve2)
            set(DUCC0_PROFILE_FLAGS_sve2
                "SHELL:-march=armv8-a+sve2 -msve-vector-bits=scalable")
            math(EXPR DUCC0_ARM_PROFILE_MASK "${DUCC0_ARM_PROFILE_MASK} | 8")
            list(APPEND DUCC0_ARM_MODULE_DEFS DUCC0_ARM_BUILD_SVE2)
            message(STATUS "ARM64 multiarch: scalable SVE2 profile enabled")
        else()
            message(STATUS
                "ARM64 multiarch: SVE2 profile disabled; compiler rejected -march=armv8-a+sve2 -msve-vector-bits=scalable")
        endif()
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_QUIET)
        list(APPEND DUCC0_ARM_MODULE_DEFS
            DUCC0_ARM_COMPILED_PROFILES_MASK=${DUCC0_ARM_PROFILE_MASK})
        message(STATUS "ARM64 multiarch compiled profile mask: ${DUCC0_ARM_PROFILE_MASK}")
    endif()

    foreach(profile IN LISTS MULTIARCH_PROFILES)
        add_library(${PKGNAME}_lib_${profile} OBJECT
            ${PYMOD_SOURCES} ${CXX_SOURCES})
        set_property(TARGET ${PKGNAME}_lib_${profile}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    endforeach()
else()
    add_library(${PKGNAME}_lib OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
endif()

if (DEFINED ENV{DUCC0_USE_NANOBIND})
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_DIR)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_INCDIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    set(INC_DIRS ${INC_DIRS} ${NB_INCDIR} ${Python_INCLUDE_DIRS})
    if (MULTIARCH)
        foreach(profile IN LISTS MULTIARCH_PROFILES)
            nanobind_compile_options(${PKGNAME}_lib_${profile})
        endforeach()
    else()
        nanobind_compile_options(${PKGNAME}_lib)
    endif()
    nanobind_add_module(${PKGNAME}
        FREE_THREADED NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE
        python/ducc_driver.cc python/multiarch.cc)
    set (DEFS ${DEFS} DUCC0_USE_NANOBIND)
# Enable this if stubs are required
#    nanobind_add_stub(
#        ${PKGNAME}_stub
#        MODULE ${PKGNAME}
#        OUTPUT ${PKGNAME}.pyi
#        PYTHON_PATH $<TARGET_FILE_DIR:${PKGNAME}>
#        DEPENDS ${PKGNAME}
#        RECURSIVE
#    )
else()
    find_package(pybind11 REQUIRED)
    set(INC_DIRS ${INC_DIRS} ${pybind11_INCLUDE_DIRS})
    pybind11_add_module(${PKGNAME} NO_EXTRAS
        python/ducc_driver.cc python/multiarch.cc)
endif()

if (NOT MSVC)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(DUCC0_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (CMAKE_SYSTEM_NAME MATCHES "Emscripten")
        set(DUCC0_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    if (OPTFLAGS MATCHES "^native")
        message(STATUS "Using GCC/Clang flags: ${DUCC0_ARCH_FLAGS}")
    elseif (MULTIARCH)
        message(STATUS
            "Multiarch mode uses explicit profile architecture flags; DUCC0_ARCH_FLAGS is not applied")
    endif()
else ()
    # Check for AVX, AVX512 and SSE support
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
            RUN_OUTPUT_VARIABLE RUN_OUTPUT)
    if (RUN_OUTPUT MATCHES "AVX512")
        set(DUCC0_ARCH_FLAGS "/arch:AVX512" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "AVX")
        set(DUCC0_ARCH_FLAGS "/arch:AVX" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "SSE")
        set(DUCC0_ARCH_FLAGS "/arch:SSE" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${DUCC0_ARCH_FLAGS}")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(DUCC0_WARN_FLAGS
        -Wfatal-errors -Wfloat-conversion -W -Wall -Wstrict-aliasing
        -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual
        -Wcast-align -Wpointer-arith -Wnon-virtual-dtor
        -Wzero-as-null-pointer-constant -Wno-psabi)
#    set (DUCC0_OPT_FLAGS -O3 -ffp-contract=fast -fexcess-precision=fast -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math)
    # we can use -ffast-math here because the flags will not be used for linking
    set (DUCC0_OPT_FLAGS -O3 -ffast-math)
    set (DUCC0_NOOPT_FLAGS -O0)
    set (DUCC0_DEBUG_FLAGS -g)
    set (DUCC0_STRIP_FLAGS -s)
endif()
SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_WARN_FLAGS})

if (OPTFLAGS MATCHES "^native")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_ARCH_FLAGS})
endif()
if (OPTFLAGS MATCHES "-debug$")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_DEBUG_FLAGS})
elseif (OPTFLAGS MATCHES "-strip$")
    SET (LINK_OPTS ${LINK_OPTS} ${DUCC0_STRIP_FLAGS})
endif()
if (OPTFLAGS MATCHES "^none")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_NOOPT_FLAGS})
else()
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_OPT_FLAGS})
endif()

if(DEFINED ENV{DUCC0_CFLAGS})
    separate_arguments(CFLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_CFLAGS}")
    SET (COMP_OPTS ${COMP_OPTS} ${CFLAGS_TMP})
endif()
if(DEFINED ENV{DUCC0_LFLAGS})
    separate_arguments(LFLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_LFLAGS}")
    SET (LINK_OPTS ${LINK_OPTS} ${LFLAGS_TMP})
endif()
if(DEFINED ENV{DUCC0_FLAGS})
    separate_arguments(FLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_FLAGS}")
    SET (COMP_OPTS ${COMP_OPTS} ${FLAGS_TMP})
    SET (LINK_OPTS ${LINK_OPTS} ${FLAGS_TMP})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Emscripten")
    SET (DEFS ${DEFS} DUCC0_NO_LOWLEVEL_THREADING)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PKGNAME} PRIVATE Threads::Threads)
endif()

SET (DEFS ${DEFS} PKGVERSION=${PKGVERSION})

# settings for the final compile/link step
target_compile_definitions(${PKGNAME} PRIVATE
    ${DEFS} ${DUCC0_ARM_MODULE_DEFS} PKGNAME=${PKGNAME})
if (MULTIARCH_ARCH STREQUAL "arm64"
    AND DEFINED ENV{DUCC0_USE_NANOBIND})
    # nanobind's core library is a separate target, so give it the same
    # conservative baseline and LTO isolation as the module driver.
    foreach(nb_target nanobind-static nanobind-static-ft)
        if (TARGET ${nb_target})
            set_property(TARGET ${nb_target}
                PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
            target_compile_options(${nb_target} PRIVATE
                ${DUCC0_BASELINE_FLAGS} ${DUCC0_ISOLATION_FLAGS})
        endif()
    endforeach()
endif()
if (MULTIARCH)
    if (MULTIARCH_ARCH STREQUAL "arm64")
        set_property(TARGET ${PKGNAME}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
        target_compile_options(${PKGNAME} PRIVATE
            ${COMP_OPTS} ${DUCC0_BASELINE_FLAGS} ${DUCC0_ISOLATION_FLAGS})
    else()
        target_compile_options(${PKGNAME} PRIVATE
            ${COMP_OPTS} ${DUCC0_BASELINE_FLAGS})
    endif()
else()
    target_compile_options(${PKGNAME} PRIVATE ${COMP_OPTS})
endif()
if (MULTIARCH AND MULTIARCH_ARCH STREQUAL "arm64")
    target_link_options(${PKGNAME} PRIVATE ${LINK_OPTS} ${DUCC0_ISOLATION_FLAGS})
elseif (MULTIARCH)
    target_link_options(${PKGNAME} PRIVATE ${LINK_OPTS})
else()
    target_link_options(${PKGNAME} PRIVATE ${LINK_OPTS})
endif()

# setting s for the library part(s)
if (MULTIARCH)
    foreach(profile IN LISTS MULTIARCH_PROFILES)
        set(profile_var "DUCC0_PROFILE_FLAGS_${profile}")
        target_include_directories(${PKGNAME}_lib_${profile}
            SYSTEM PRIVATE ${INC_DIRS})
        target_compile_definitions(${PKGNAME}_lib_${profile} PRIVATE
            ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_${profile})
        target_compile_options(${PKGNAME}_lib_${profile} PRIVATE
            ${COMP_OPTS} ${${profile_var}} ${DUCC0_ISOLATION_FLAGS})
        target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib_${profile})
    endforeach()
else()
    target_include_directories(${PKGNAME}_lib SYSTEM PRIVATE ${INC_DIRS})
    target_compile_definitions(${PKGNAME}_lib PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME})
    target_compile_options(${PKGNAME}_lib PRIVATE ${COMP_OPTS})
    target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib)
endif()

install(TARGETS ${PKGNAME} LIBRARY DESTINATION .)
