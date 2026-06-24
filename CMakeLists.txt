####################################################
############     LOOPS CMAKE CONFIG     ############
####################################################

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

# ----- CUDA architecture selection ----------------------------------------
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
message(STATUS "Loops CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ----- Toolkit & system dependencies --------------------------------------
# Provides imported targets: CUDA::cudart, CUDA::cuda_driver, CUDA::curand, ...
# CUDAToolkit is required (>= 11.7) and supplies the bundled Thrust + CUB
# headers via CUDAToolkit_INCLUDE_DIRS, so we no longer fetch them separately.
find_package(CUDAToolkit 11.7 REQUIRED)
message(STATUS "Found CUDA Toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")

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

if(NOT LOOPS_USE_BUNDLED_CCCL)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCCCL.cmake)
endif()

# ----- The Loops library (header-only) ------------------------------------
add_library(loops INTERFACE)
add_library(loops::loops ALIAS loops)

target_compile_features(loops INTERFACE cxx_std_17 cuda_std_17)

target_compile_definitions(loops
  INTERFACE
    LOOPS_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    LOOPS_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    LOOPS_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Bake the primary target arch into the kernels so the launch_box (util/
# launch_box.hxx) resolves block size / items-per-thread to the GPU at compile
# time. A single-arch preset (release-a100 -> 80, release-h100 -> 90) pins it
# exactly; multi-arch / native builds leave it at the Ampere floor and rely on
# each box's fallback entry plus the run-time grid sizing.
list(LENGTH CMAKE_CUDA_ARCHITECTURES _loops_arch_count)
if(_loops_arch_count EQUAL 1 AND CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
  target_compile_definitions(loops
    INTERFACE LOOPS_TARGET_ARCH=${CMAKE_CUDA_ARCHITECTURES})
  message(STATUS "Loops compile-time SpMV tuning: sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

target_include_directories(loops
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CXXOPTS_INCLUDE_DIR}
)

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

target_compile_options(loops INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${LOOPS_HOST_WARNINGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${LOOPS_CUDA_DEBUG_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:${LOOPS_CUDA_RELEASE_FLAGS}>
)

# Examples / tests / benchmarks need separable compilation + propagation of
# CUDA_ARCHITECTURES from the global default. Set them via a helper so callers
# don't have to repeat the boilerplate.
function(loops_setup_cuda_target target)
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
