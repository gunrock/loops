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
cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

# begin /* Update Loops version */
set(LOOPS_VERSION_MAJOR 2)
set(LOOPS_VERSION_MINOR 0)
set(LOOPS_VERSION_PATCH 0)
# end /* Update Loops version */

# Adding CMAKE_PREFIX_PATH, needed for static builds
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)


set(LOOPS_VERSION "${LOOPS_VERSION_MAJOR}.${LOOPS_VERSION_MINOR}.${LOOPS_VERSION_PATCH}")

project(loops
  VERSION ${LOOPS_VERSION}
  LANGUAGES CXX C HIP
)

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
include(${PROJECT_SOURCE_DIR}/cmake/FetchHIP.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchColors.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThrustCUB.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchModernGPU.cmake)
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
############### SET SM ARCHITECTURE ################
####################################################

## Note: This applies to NVBench as well.
## Can be used for applications by extracting the
## HIP_ARCHITECTURES property from loops project.
## see: get_target_properties()
# message(STATUS "${Magenta}Set SM Architecture using -DCMAKE_HIP_ARCHITECTURES=75,80${ColourReset}")
set(CMAKE_HIP_ARCHITECTURES gfx900)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
set_target_properties(loops 
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF # Should this be turned on for MSVC?
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    HIP_EXTENSIONS OFF
    # HIP_RESOLVE_DEVICE_SYMBOLS ON
    # HIP_SEPARABLE_COMPILATION ON
    HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}
    # HIP_PTX_COMPILATION ON # Can only be applied to OBJ.
)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
target_compile_definitions(loops
  INTERFACE 
    SM_TARGET=${CMAKE_HIP_ARCHITECTURES}
    LOOPS_VERSION=${LOOPS_VERSION}
    __HIP_PLATFORM_AMD__
)

message(STATUS "Loops HIP Architecture: ${CMAKE_HIP_ARCHITECTURES}")

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
  INTERFACE /opt/rocm/include
  INTERFACE /opt/rocm/hip/include/
  # INTERFACE ${MODERNGPU_INCLUDE_DIR}
  # INTERFACE ${RAPIDJSON_INCLUDE_DIR}
  # INTERFACE ${CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES}
)

####################################################
############ TARGET LINK LIBRARIES #################
####################################################
target_link_directories(loops 
  INTERFACE /opt/rocm/lib
)

target_link_libraries(loops
  # INTERFACE hip
  INTERFACE rocrand
)

####################################################
################# TARGET SOURCES ###################
####################################################
# target_sources(loops
#   INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/container/detail/mmio.cpp"
# )

####################################################
############## SET CXX & HIP FLAGS ################
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

set(HIP_FLAGS
  # --expt-extended-lambda 
  # --expt-relaxed-constexpr
  # --use_fast_math
  # --ptxas-options -v
  # --generate-line-info
  --offload-arch=gfx900
  $<$<CXX_COMPILER_ID:GNU>:-O3> # Host optimize-level
  # --verbose
  # --debug # Host debug
  # --device-debug # Device debug
)

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################
target_compile_options(loops INTERFACE 
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>
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
