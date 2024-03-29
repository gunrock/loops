####################################################
############ INSTALLING CORRECT CMAKE ##############
####################################################
# Installing correct cmake version is easy!
# 1) Find the respective version here;
# https://github.com/Kitware/CMake/releases,
# and 2) replace the [x.xx.x] in the following
# commands with the version number (remove the
# brackets). For example, if you are installing
# CMake 3.22.1, replace [x.xx.x] with 3.22.1:

# wget https://github.com/Kitware/CMake/releases/download/v[x.xx.x]/cmake-[x.xx.x]-linux-x86_64.sh
# chmod +x ./cmake-[x.xx.x]-linux-x86_64.sh
# ./cmake-[x.xx.x]-linux-x86_64.sh
# sudo mv cmake-[x.xx.x]-linux-x86_64 /opt/cmake
# sudo ln -s /opt/cmake/bin/* /usr/local/bin/
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Select "Release" as the default build type.
# This can be altered by setting -DCMAKE_BUILD_TYPE
# in the command-line interface to Release or Debug.
# No reason to set CMAKE_CONFIGURATION_TYPES if it's 
# not a multiconfig generator. Also no reason mess 
# with CMAKE_BUILD_TYPE if it's a multiconfig generator.
# https://stackoverflow.com/a/31548693/5729690
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
else()
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Defaulting to Release build type")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    # set the valid options for cmake-gui drop-down list
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
endif()

####################################################
############### SET SM ARCHITECTURE ################
####################################################

## Note: This applies to NVBench as well.
## Can be used for applications by extracting the
## CUDA_ARCHITECTURES property from loops project.
## see: get_target_properties()
## see: https://github.com/RadeonOpenCompute/rocminfo/blob/master/rocm_agent_enumerator#L12
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  message(STATUS "Using default GPU Architecture (CMAKE_CUDA_ARCHITECTURES): ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "GPU Architecture (CMAKE_CUDA_ARCHITECTURES): ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(loops
  LANGUAGES CXX C CUDA
)

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
include(${PROJECT_SOURCE_DIR}/cmake/FetchColors.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThrustCUB.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchCXXOpts.cmake)
# end /* Include cmake modules */

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored")

## Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

############ ADD LIBRARY: LOOPS (HEADER-ONLY) ############
add_library(loops INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
set_target_properties(loops 
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF # Should this be turned on for MSVC?
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    # CUDA_PTX_COMPILATION ON # Can only be applied to OBJ.
)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
target_compile_definitions(loops
  INTERFACE 
    SM_TARGET=${CMAKE_CUDA_ARCHITECTURES}
    LOOPS_VERSION=${LOOPS_VERSION}
)

message(STATUS "Loops CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(loops INTERFACE cxx_std_17)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(LOOPS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
target_include_directories(loops
  INTERFACE ${LOOPS_INCLUDE_DIR}
  INTERFACE ${CXXOPTS_INCLUDE_DIR}
  INTERFACE ${CUB_INCLUDE_DIR}
  INTERFACE ${THRUST_INCLUDE_DIR}
  INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

####################################################
############ TARGET LINK LIBRARIES #################
####################################################
target_link_libraries(loops
  INTERFACE curand
  INTERFACE cuda
)

####################################################
################# TARGET SOURCES ###################
####################################################
target_sources(loops
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/container/detail/mmio.cpp"
)

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################
set(CXX_FLAGS 
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4
  >
  $<$<CXX_COMPILER_ID:GNU>:
    -Wall 
    # -Wextra
    -Wno-unused-result
    -Wno-unused-local-typedefs
    -Wno-strict-aliasing
    -Wno-unused-function
    -Wno-format-security
    # -Werror
    # -vvv
  >
)

set(CUDA_RELEASE_FLAGS
  --expt-extended-lambda 
  --expt-relaxed-constexpr
  --use_fast_math
)

set(CUDA_DEBUG_FLAGS
  --expt-extended-lambda 
  --expt-relaxed-constexpr
  --ptxas-options -v
  --debug # Host debug
  --device-debug # Device debug
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################
target_compile_options(loops INTERFACE 
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CXX_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:${CUDA_DEBUG_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:${CUDA_RELEASE_FLAGS}>
)

####################################################
############ BUILD EXAMPLE APPLICATIONS ############
####################################################
option(LOOPS_BUILD_EXAMPLES
  "If on, builds the example applications."
  ON)

# Subdirectories for examples, testing and documentation
if(LOOPS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(LOOPS_BUILD_EXAMPLES)

####################################################
################ BUILD UNIT TESTS  #################
####################################################
option(LOOPS_BUILD_TESTS
  "If on, builds the unit tests."
  OFF)

# Subdirectories for examples, testing and documentation
if(LOOPS_BUILD_TESTS)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchGoogleTest.cmake)
  # add_subdirectory(unittests)
endif(LOOPS_BUILD_TESTS)

####################################################
################ BUILD BENCHMARKS  #################
####################################################
option(LOOPS_BUILD_BENCHMARKS
  "If on, builds loops with benchmarking support."
  OFF)

# Subdirectories for examples, testing and documentation
if(LOOPS_BUILD_BENCHMARKS)
  # ... see https://github.com/NVIDIA/nvbench/issues/66
  set(NVBench_ENABLE_NVML OFF)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNVBench.cmake)
  add_subdirectory(benchmarks)
endif(LOOPS_BUILD_BENCHMARKS)