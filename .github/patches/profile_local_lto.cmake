# Benchmark-only multiarch link configuration. Included by the temporary
# CMakeLists.txt patch made in each isolated source worktree.
function(ducc0_bench_configure_multiarch_link module)
  set_property(TARGET ${module} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_compile_options(${module} PRIVATE -fno-lto)
  target_link_options(${module} PRIVATE -fno-lto)

  if(DUCC0_PROFILE_LOCAL_LTO)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      message(FATAL_ERROR "Profile-local LTO requires GCC; no fallback is allowed")
    endif()

    foreach(profile IN ITEMS v1 v3 v4)
      set(object_target ${module}_lib_${profile})
      set_property(TARGET ${object_target}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

      if(profile STREQUAL "v1")
        set(march x86-64)
      elseif(profile STREQUAL "v3")
        set(march x86-64-v3)
      else()
        set(march x86-64-v4)
      endif()

      set(native_object "${CMAKE_CURRENT_BINARY_DIR}/${module}_${profile}.native.o")
      add_custom_command(
        OUTPUT "${native_object}"
        COMMAND "${CMAKE_CXX_COMPILER}"
          -r -O3 -fPIC -flto -flto-partition=none
          -flinker-output=nolto-rel "-march=${march}"
          -o "${native_object}" "$<TARGET_OBJECTS:${object_target}>"
        DEPENDS ${object_target}
        COMMAND_EXPAND_LISTS
        VERBATIM
        COMMENT "Creating profile-local native ${profile} relocatable object")
      set_source_files_properties("${native_object}"
        PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
      target_sources(${module} PRIVATE "${native_object}")
      add_custom_target(${module}_${profile}_native
        DEPENDS "${native_object}")
      add_dependencies(${module} ${module}_${profile}_native)
    endforeach()
  else()
    foreach(profile IN ITEMS v1 v3 v4)
      target_link_libraries(${module} PRIVATE ${module}_lib_${profile})
    endforeach()
  endif()
endfunction()
