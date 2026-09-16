# Runtime-dispatched SHT target configuration for DUCC's x86-64 build.

function(_ducc0_cpu_dispatch_validate_global_flags)
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION OR
       CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG OR
       CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OR
       CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OR
       CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL)
        message(FATAL_ERROR
            "DUCC0_CPU_DISPATCH cannot be combined with IPO/LTO flags; "
            "disable CMake interprocedural optimization for the dispatch build")
    endif()

    set(_ducc0_lto_pattern
        "(^|[ ;])(-flto([=][^ ;]*)?|/GL|/LTCG)([ ;]|$)")
    foreach(_ducc0_flags
        "${CMAKE_CXX_FLAGS}"
        "${CMAKE_CXX_FLAGS_DEBUG}"
        "${CMAKE_CXX_FLAGS_RELEASE}"
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_CXX_FLAGS_MINSIZEREL}"
        "${CMAKE_EXE_LINKER_FLAGS}"
        "${CMAKE_EXE_LINKER_FLAGS_DEBUG}"
        "${CMAKE_EXE_LINKER_FLAGS_RELEASE}"
        "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL}"
        "${CMAKE_SHARED_LINKER_FLAGS}"
        "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}"
        "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}"
        "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL}"
        "${CMAKE_MODULE_LINKER_FLAGS}"
        "${CMAKE_MODULE_LINKER_FLAGS_DEBUG}"
        "${CMAKE_MODULE_LINKER_FLAGS_RELEASE}"
        "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL}"
        "$ENV{CXXFLAGS}"
        "$ENV{DUCC0_CFLAGS}"
        "$ENV{DUCC0_LFLAGS}"
        "$ENV{DUCC0_FLAGS}")
        if(_ducc0_flags MATCHES "${_ducc0_lto_pattern}")
            message(FATAL_ERROR
                "DUCC0_CPU_DISPATCH cannot be combined with IPO/LTO flags; "
                "remove -flto, /GL, or /LTCG from the dispatch build")
        endif()
    endforeach()

    # These options can change the instructions emitted by every target. The
    # target-local profiles below are the only permitted ISA selectors here.
    set(_ducc0_isa_pattern
        "(^|[ ;])(-m(no-)?(arch|cpu|avx|sse|ssse|fma|bmi|fma4|xop|f16c|aes|pclmul|popcnt|sha|amx|abm|lzcnt|tbm|mmx|3dnow)[^ ;]*|-x(H|h)ost[^ ;]*|-x(SSE|AVX|CORE-AVX)[^ ;]*|/arch:[^ ;]+)([ ;]|$)")
    foreach(_ducc0_flags
        "${CMAKE_CXX_FLAGS}"
        "${CMAKE_CXX_FLAGS_DEBUG}"
        "${CMAKE_CXX_FLAGS_RELEASE}"
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
        "${CMAKE_CXX_FLAGS_MINSIZEREL}"
        "$ENV{CXXFLAGS}"
        "$ENV{DUCC0_CFLAGS}"
        "$ENV{DUCC0_FLAGS}")
        if(_ducc0_flags MATCHES "${_ducc0_isa_pattern}")
            message(FATAL_ERROR
                "DUCC0_CPU_DISPATCH cannot guarantee a neutral x86-64 target "
                "with a global ISA flag; remove explicit architecture/SIMD "
                "flags from CXXFLAGS, DUCC0_CFLAGS, or DUCC0_FLAGS and use "
                "the target-local dispatch flags")
        endif()
    endforeach()
endfunction()

function(_ducc0_cpu_dispatch_set_inactive _ducc0_reason)
    set(DUCC0_CPU_DISPATCH_ACTIVE OFF CACHE INTERNAL
        "Whether DUCC runtime CPU dispatch is active" FORCE)
    set(DUCC0_CPU_DISPATCH_ACTIVE OFF PARENT_SCOPE)
    if(NOT "${_ducc0_reason}" STREQUAL "")
        message(WARNING
            "${_ducc0_reason}; retaining the normal compile-time SIMD build")
    endif()
endfunction()

function(ducc0_configure_cpu_dispatch)
    set(DUCC0_CPU_DISPATCH_ACTIVE OFF CACHE INTERNAL
        "Whether DUCC runtime CPU dispatch is active" FORCE)
    set(DUCC0_CPU_DISPATCH_ACTIVE OFF PARENT_SCOPE)

    if(NOT DUCC0_CPU_DISPATCH)
        return()
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ducc0_processor)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _ducc0_generator_platform)
    set(_ducc0_is_x86_64 OFF)
    if(_ducc0_processor MATCHES "^(x86_64|amd64|x64)$" OR
       _ducc0_generator_platform MATCHES "^(x64|amd64)$")
        set(_ducc0_is_x86_64 ON)
    endif()
    if(_ducc0_processor MATCHES "^(aarch64|arm64|armv|arm)" OR
       _ducc0_generator_platform MATCHES "^(arm|arm64|arm64ec)$")
        set(_ducc0_is_x86_64 OFF)
    endif()
    if(NOT _ducc0_is_x86_64)
        _ducc0_cpu_dispatch_set_inactive(
            "DUCC0_CPU_DISPATCH is supported only on x86-64 targets")
        return()
    endif()

    # clang-cl has an MSVC-compatible frontend but the GNU-style profile below
    # is intentionally not a clang-cl compatibility layer.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
       (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC" OR
        CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
        _ducc0_cpu_dispatch_set_inactive(
            "DUCC0_CPU_DISPATCH does not support an MSVC-compatible Clang frontend")
        return()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(_ducc0_sse2_flags /arch:SSE2)
        set(_ducc0_avx_flags /arch:AVX)
        set(_ducc0_avx512_flags /arch:AVX512)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
           CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(_ducc0_sse2_flags -march=x86-64)
        set(_ducc0_avx_flags -mavx)
        set(_ducc0_avx512_flags -mavx512f)
    else()
        _ducc0_cpu_dispatch_set_inactive(
            "DUCC0_CPU_DISPATCH has no target profile for compiler ${CMAKE_CXX_COMPILER_ID}")
        return()
    endif()

    _ducc0_cpu_dispatch_validate_global_flags()

    string(JOIN " " _ducc0_sse2_probe_flags ${_ducc0_sse2_flags})
    string(JOIN " " _ducc0_avx_probe_flags
        ${_ducc0_sse2_flags} ${_ducc0_avx_flags})
    string(JOIN " " _ducc0_avx512_probe_flags
        ${_ducc0_sse2_flags} ${_ducc0_avx512_flags})

    include(CheckCXXSourceCompiles)
    if(DEFINED CMAKE_REQUIRED_FLAGS)
        set(_ducc0_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
        set(_ducc0_required_flags_was_defined ON)
    else()
        set(_ducc0_required_flags_was_defined OFF)
    endif()
    if(DEFINED CMAKE_TRY_COMPILE_TARGET_TYPE)
        set(_ducc0_saved_try_compile_target_type
            "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
        set(_ducc0_try_compile_type_was_defined ON)
    else()
        set(_ducc0_try_compile_type_was_defined OFF)
    endif()

    # These are compile-only capability checks. STATIC_LIBRARY avoids a
    # configure-time executable link or execution step.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

    unset(DUCC0_CPU_DISPATCH_CHECK_SSE2 CACHE)
    unset(DUCC0_CPU_DISPATCH_CHECK_AVX CACHE)
    unset(DUCC0_CPU_DISPATCH_CHECK_AVX512 CACHE)

    set(CMAKE_REQUIRED_FLAGS "${_ducc0_sse2_probe_flags}")
    check_cxx_source_compiles(
        "#include <emmintrin.h>\nint main() { double values[2]; _mm_storeu_pd(values, _mm_setzero_pd()); return values[0] != 0.; }"
        DUCC0_CPU_DISPATCH_CHECK_SSE2)
    if(NOT DUCC0_CPU_DISPATCH_CHECK_SSE2)
        if(_ducc0_try_compile_type_was_defined)
            set(CMAKE_TRY_COMPILE_TARGET_TYPE
                "${_ducc0_saved_try_compile_target_type}")
        else()
            unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
        endif()
        if(_ducc0_required_flags_was_defined)
            set(CMAKE_REQUIRED_FLAGS "${_ducc0_saved_required_flags}")
        else()
            unset(CMAKE_REQUIRED_FLAGS)
        endif()
        message(FATAL_ERROR
            "The compiler cannot emit the required x86-64 SSE2 target")
    endif()

    set(CMAKE_REQUIRED_FLAGS "${_ducc0_avx_probe_flags}")
    check_cxx_source_compiles(
        "#include <immintrin.h>\nint main() { double values[4]; _mm256_storeu_pd(values, _mm256_setzero_pd()); return values[0] != 0.; }"
        DUCC0_CPU_DISPATCH_CHECK_AVX)

    set(CMAKE_REQUIRED_FLAGS "${_ducc0_avx512_probe_flags}")
    check_cxx_source_compiles(
        "#include <immintrin.h>\nint main() { double values[8]; _mm512_storeu_pd(values, _mm512_setzero_pd()); return values[0] != 0.; }"
        DUCC0_CPU_DISPATCH_CHECK_AVX512)

    if(_ducc0_required_flags_was_defined)
        set(CMAKE_REQUIRED_FLAGS "${_ducc0_saved_required_flags}")
    else()
        unset(CMAKE_REQUIRED_FLAGS)
    endif()
    if(_ducc0_try_compile_type_was_defined)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE
            "${_ducc0_saved_try_compile_target_type}")
    else()
        unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    endif()

    set(DUCC0_CPU_DISPATCH_SSE2_FLAGS "${_ducc0_sse2_flags}"
        CACHE INTERNAL "DUCC dispatch SSE2 target flags" FORCE)
    set(DUCC0_CPU_DISPATCH_AVX_FLAGS "${_ducc0_avx_flags}"
        CACHE INTERNAL "DUCC dispatch AVX target flags" FORCE)
    set(DUCC0_CPU_DISPATCH_AVX512_FLAGS "${_ducc0_avx512_flags}"
        CACHE INTERNAL "DUCC dispatch AVX512 target flags" FORCE)
    set(DUCC0_CPU_DISPATCH_COMPILER_HAS_AVX
        "${DUCC0_CPU_DISPATCH_CHECK_AVX}" CACHE INTERNAL
        "Whether the compiler can build DUCC's AVX target" FORCE)
    set(DUCC0_CPU_DISPATCH_COMPILER_HAS_AVX512
        "${DUCC0_CPU_DISPATCH_CHECK_AVX512}" CACHE INTERNAL
        "Whether the compiler can build DUCC's AVX512 target" FORCE)

    set(DUCC0_CPU_DISPATCH_ACTIVE ON CACHE INTERNAL
        "Whether DUCC runtime CPU dispatch is active" FORCE)
    set(DUCC0_CPU_DISPATCH_ACTIVE ON PARENT_SCOPE)

    message(STATUS "CPU dispatch: enabled for x86-64")
    message(STATUS "CPU dispatch SSE2 target: ${DUCC0_CPU_DISPATCH_CHECK_SSE2}")
    message(STATUS "CPU dispatch AVX target: ${DUCC0_CPU_DISPATCH_CHECK_AVX}")
    message(STATUS "CPU dispatch AVX512 target: ${DUCC0_CPU_DISPATCH_CHECK_AVX512}")
endfunction()

function(_ducc0_cpu_dispatch_add_sht_target _ducc0_target_name _ducc0_target_id)
    set(_ducc0_target_flags ${ARGN})
    add_library(${_ducc0_target_name} OBJECT
        "${CMAKE_SOURCE_DIR}/src/ducc0/sht/sht_dispatch_target.cc")
    target_include_directories(${_ducc0_target_name} PRIVATE
        "${CMAKE_SOURCE_DIR}/src")
    target_compile_features(${_ducc0_target_name} PRIVATE cxx_std_17)
    target_compile_definitions(${_ducc0_target_name} PRIVATE
        DUCC0_CPU_DISPATCH=1
        DUCC0_DISPATCH_SHT_TARGET=${_ducc0_target_id})
    target_compile_options(${_ducc0_target_name} PRIVATE
        ${DUCC0_CPU_DISPATCH_COMMON_OPTIONS}
        ${DUCC0_CPU_DISPATCH_SSE2_FLAGS}
        ${_ducc0_target_flags})
    set_property(TARGET ${_ducc0_target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${_ducc0_target_name}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_target_properties(${_ducc0_target_name} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    target_sources(${DUCC0_CPU_DISPATCH_PARENT_TARGET} PRIVATE
        $<TARGET_OBJECTS:${_ducc0_target_name}>)
endfunction()

function(ducc0_attach_cpu_dispatch _ducc0_parent_target)
    if(NOT DUCC0_CPU_DISPATCH_ACTIVE)
        return()
    endif()

    get_target_property(DUCC0_CPU_DISPATCH_COMMON_OPTIONS
        ${_ducc0_parent_target} COMPILE_OPTIONS)
    if(DUCC0_CPU_DISPATCH_COMMON_OPTIONS STREQUAL
       "DUCC0_CPU_DISPATCH_COMMON_OPTIONS-NOTFOUND")
        set(DUCC0_CPU_DISPATCH_COMMON_OPTIONS)
    endif()
    set(DUCC0_CPU_DISPATCH_PARENT_TARGET "${_ducc0_parent_target}")

    set_property(TARGET ${_ducc0_parent_target}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_compile_definitions(${_ducc0_parent_target} PRIVATE
        DUCC0_CPU_DISPATCH=1)
    # Keep all neutral sources explicitly at the x86-64/SSE2 profile.
    target_compile_options(${_ducc0_parent_target} PRIVATE
        ${DUCC0_CPU_DISPATCH_SSE2_FLAGS})

    _ducc0_cpu_dispatch_add_sht_target(ducc0_sht_sse2 0)

    if(DUCC0_CPU_DISPATCH_COMPILER_HAS_AVX)
        target_compile_definitions(${_ducc0_parent_target} PRIVATE
            DUCC0_DISPATCH_HAS_AVX=1)
        _ducc0_cpu_dispatch_add_sht_target(ducc0_sht_avx 1
            ${DUCC0_CPU_DISPATCH_AVX_FLAGS})
    endif()

    if(DUCC0_CPU_DISPATCH_COMPILER_HAS_AVX512)
        target_compile_definitions(${_ducc0_parent_target} PRIVATE
            DUCC0_DISPATCH_HAS_AVX512=1)
        _ducc0_cpu_dispatch_add_sht_target(ducc0_sht_avx512 2
            ${DUCC0_CPU_DISPATCH_AVX512_FLAGS})
    endif()
endfunction()
