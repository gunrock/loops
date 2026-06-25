####################################################
############     LOOPS CMAKE CONFIG     ############
####################################################

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# ----- Device backend selection -------------------------------------------
# loops talks to the device runtime through loops::xpu (include/loops/util/
# xpu.hxx), so the whole tree compiles on either vendor by flipping one switch:
#   -DLOOPS_BACKEND=CUDA  (nvcc, default; existing builds unaffected)
#   -DLOOPS_BACKEND=HIP   (hipcc / AMD ROCm)
set(LOOPS_BACKEND "CUDA" CACHE STRING "Device backend: CUDA (nvcc) or HIP (hipcc/ROCm)")
set_property(CACHE LOOPS_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${LOOPS_BACKEND}" LOOPS_BACKEND)

# ----- Build type ---------------------------------------------------------
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
else()
  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Defaulting to Release build type")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug;Release;RelWithDebInfo;MinSizeRel")
endif()

if(LOOPS_BACKEND STREQUAL "HIP")
  # ----- AMD architecture selection ---------------------------------------
  # Pick gfx target(s) (gfx90a MI210/MI250, gfx942 MI300/MI325, gfx950 MI350).
  # A single-arch preset pins the compile-time launch_box tuning exactly;
  # multi-arch builds fall back to the box's fallback entry.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx942"
        CACHE STRING "AMD gfx architectures")
  endif()

  project(loops
    VERSION   0.2.0
    LANGUAGES CXX C HIP
    DESCRIPTION "An open-source GPU load-balancing framework"
  )

  message(STATUS "Loops version: ${PROJECT_VERSION}")
  message(STATUS "Loops backend: HIP (ROCm)")
  message(STATUS "Loops HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

  # rocThrust / rocPRIM provide the thrust:: API on AMD; hip::host the runtime.
  find_package(hip REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(rocthrust REQUIRED)
  message(STATUS "Found HIP ${hip_VERSION}")
else()
  # ----- CUDA architecture selection --------------------------------------
  # Default: detect the host GPU(s) (CMake 3.24+ supports `native`).
  # Override with -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" for multi-arch builds,
  # or pick a single arch like 90 for an H100-only build.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native"
        CACHE STRING "CUDA SM architectures (default: native auto-detect)")
  endif()

  project(loops
    VERSION   0.2.0
    LANGUAGES CXX C CUDA
    DESCRIPTION "An open-source GPU load-balancing framework"
  )

  message(STATUS "Loops version: ${PROJECT_VERSION}")
  message(STATUS "Loops backend: CUDA")
  message(STATUS "Loops CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  # ----- Toolkit & system dependencies ------------------------------------
  # Provides imported targets: CUDA::cudart, CUDA::cuda_driver, CUDA::curand, ...
  # CUDAToolkit is required (>= 11.7) and supplies the bundled Thrust + CUB
  # headers via CUDAToolkit_INCLUDE_DIRS, so we no longer fetch them separately.
  find_package(CUDAToolkit 11.7 REQUIRED)
  message(STATUS "Found CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")
endif()

# ----- Output directories -------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin"
    CACHE PATH "Directory where all executables will be stored")
set(LIBRARY_OUTPUT_PATH    "${PROJECT_BINARY_DIR}/lib"
    CACHE PATH "Directory where all libraries will be stored")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE        OFF)

# ----- Project options ----------------------------------------------------
option(LOOPS_BUILD_EXAMPLES   "Build example applications"  ON)
option(LOOPS_BUILD_TESTS      "Build unit tests"            OFF)
option(LOOPS_BUILD_BENCHMARKS "Build NVBench benchmarks"    OFF)
option(LOOPS_USE_BUNDLED_CCCL "Use CUDA Toolkit's bundled Thrust/CUB instead of fetching CCCL" ON)

# ----- External dependencies ----------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/FetchColors.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchCXXOpts.cmake)

if(LOOPS_BACKEND STREQUAL "CUDA" AND NOT LOOPS_USE_BUNDLED_CCCL)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCCCL.cmake)
endif()

# ----- The Loops library (header-only) ------------------------------------
add_library(loops INTERFACE)
add_library(loops::loops ALIAS loops)

if(LOOPS_BACKEND STREQUAL "HIP")
  target_compile_features(loops INTERFACE cxx_std_17 hip_std_17)
else()
  target_compile_features(loops INTERFACE cxx_std_17 cuda_std_17)
endif()

target_compile_definitions(loops
  INTERFACE
    LOOPS_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    LOOPS_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    LOOPS_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Bake the primary target arch into the kernels so the launch_box (util/
# launch_box.hxx) resolves block size / items-per-thread to the GPU at compile
# time. A single-arch preset pins it exactly (release-a100 -> sm_80,
# release-mi300x -> gfx942); multi-arch / native builds leave it at the
# per-vendor floor and rely on each box's fallback entry plus run-time grid
# sizing.
if(LOOPS_BACKEND STREQUAL "HIP")
  target_compile_definitions(loops INTERFACE LOOPS_BACKEND_HIP)
  list(LENGTH CMAKE_HIP_ARCHITECTURES _loops_arch_count)
  # Map a single gfx target (e.g. gfx942) to the hex id the launch_box keys on
  # (LOOPS_TARGET_GFX=0x942); strip the leading sramecc/xnack feature suffix.
  if(_loops_arch_count EQUAL 1)
    string(REGEX REPLACE ":.*$" "" _loops_gfx "${CMAKE_HIP_ARCHITECTURES}")
    if(_loops_gfx MATCHES "^gfx([0-9a-fA-F]+)$")
      target_compile_definitions(loops
        INTERFACE LOOPS_TARGET_GFX=0x${CMAKE_MATCH_1})
      message(STATUS "Loops compile-time SpMV tuning: ${_loops_gfx}")
    endif()
  endif()
else()
  list(LENGTH CMAKE_CUDA_ARCHITECTURES _loops_arch_count)
  if(_loops_arch_count EQUAL 1 AND CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
    target_compile_definitions(loops
      INTERFACE LOOPS_TARGET_ARCH=${CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "Loops compile-time SpMV tuning: sm_${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()

target_include_directories(loops
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CXXOPTS_INCLUDE_DIR}
)

if(LOOPS_BACKEND STREQUAL "HIP")
  # rocThrust pulls in rocPRIM and the thrust:: API; hip::host the runtime.
  target_link_libraries(loops INTERFACE hip::host roc::rocthrust)
else()
  # Bundled Thrust/CUB headers come from the toolkit include dirs (CUDA >= 11.7).
  # When LOOPS_USE_BUNDLED_CCCL=OFF, FetchCCCL.cmake exposes CCCL::Thrust / CCCL::CUB.
  if(LOOPS_USE_BUNDLED_CCCL)
    target_include_directories(loops INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
  else()
    target_link_libraries(loops INTERFACE CCCL::Thrust CCCL::CUB CCCL::libcudacxx)
  endif()

  target_link_libraries(loops
    INTERFACE
      CUDA::cudart
      CUDA::curand
      CUDA::cuda_driver
  )
endif()

# ----- Compile flags ------------------------------------------------------
# Host-side warnings (CXX). NVCC forwards these via -Xcompiler.
set(LOOPS_HOST_WARNINGS
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -Wall
    -Wno-unused-result
    -Wno-unused-local-typedefs
    -Wno-strict-aliasing
    -Wno-unused-function
    -Wno-format-security
  >
)

set(LOOPS_CUDA_RELEASE_FLAGS
  --expt-extended-lambda
  --expt-relaxed-constexpr
  --use_fast_math
)

set(LOOPS_CUDA_DEBUG_FLAGS
  --expt-extended-lambda
  --expt-relaxed-constexpr
  --ptxas-options=-v
  -G
  -lineinfo
)

# HIP/clang takes extended lambdas and relaxed constexpr natively, so the
# release knob is just fast math (the nvcc --use_fast_math analog).
set(LOOPS_HIP_RELEASE_FLAGS
  -ffast-math
)

# HIP marks its runtime functions warn_unused_result; the fire-and-forget xpu
# wrappers (stream_synchronize, memcpy, ...) intentionally drop the status the
# CUDA path also ignores, so quiet that one diagnostic on the AMD compiler.
set(LOOPS_HIP_FLAGS
  -Wno-unused-result
)

target_compile_options(loops INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${LOOPS_HOST_WARNINGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${LOOPS_CUDA_DEBUG_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:${LOOPS_CUDA_RELEASE_FLAGS}>
  $<$<COMPILE_LANGUAGE:HIP>:${LOOPS_HIP_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<CONFIG:Debug>>>:${LOOPS_HIP_RELEASE_FLAGS}>
)

# Examples / tests / benchmarks need separable compilation + propagation of the
# arch list from the global default. A single helper sets the per-backend target
# properties so callers don't repeat the boilerplate; on HIP it also compiles
# the shared `.cu` sources as HIP (they are vendor-neutral through loops::xpu).
function(loops_setup_cuda_target target)
  if(LOOPS_BACKEND STREQUAL "HIP")
    get_target_property(_loops_srcs ${target} SOURCES)
    foreach(_src IN LISTS _loops_srcs)
      if(_src MATCHES "\\.cu$")
        set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
    set_target_properties(${target} PROPERTIES
      CXX_STANDARD              17
      CXX_STANDARD_REQUIRED     ON
      CXX_EXTENSIONS            OFF
      HIP_STANDARD              17
      HIP_STANDARD_REQUIRED     ON
      HIP_ARCHITECTURES         "${CMAKE_HIP_ARCHITECTURES}"
    )
  else()
    set_target_properties(${target} PROPERTIES
      CXX_STANDARD              17
      CXX_STANDARD_REQUIRED     ON
      CXX_EXTENSIONS            OFF
      CUDA_STANDARD             17
      CUDA_STANDARD_REQUIRED    ON
      CUDA_EXTENSIONS           OFF
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  endif()
endfunction()

# ----- Subdirectories -----------------------------------------------------
if(LOOPS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(LOOPS_BUILD_TESTS)
  enable_testing()
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCatch2.cmake)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/CMakeLists.txt"
     AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/unittests")
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/CMakeLists.txt" _ut_size)
    if(_ut_size GREATER 0)
      add_subdirectory(unittests)
    else()
      message(WARNING "LOOPS_BUILD_TESTS is ON but unittests/CMakeLists.txt is empty; skipping.")
    endif()
  endif()
endif()

if(LOOPS_BUILD_BENCHMARKS)
  set(NVBench_ENABLE_NVML OFF CACHE BOOL "" FORCE)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNVBench.cmake)
  add_subdirectory(benchmarks)
endif()
