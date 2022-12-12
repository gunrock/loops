include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning External Project: Rocsparse")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

set(CMAKE_CXX_COMPILER_ID "Clang")
set(BUILD_CLIENTS_SAMPLES OFF)

FetchContent_Declare(
    rocsparse
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/rocSPARSE.git
    GIT_TAG        rocm-5.4
)

FetchContent_GetProperties(rocsparse)
if(NOT rocsparse_POPULATED)
  FetchContent_Populate(
    rocsparse
  )
endif()
# Exposing rocsparse's source and include directory
set(ROCSPARSE_INCLUDE_DIR "${rocsparse_SOURCE_DIR}")
set(ROCSPARSE_BUILD_DIR "${rocsparse_BINARY_DIR}")

# Add subdirectory ::rocsparse
add_subdirectory(${ROCSPARSE_INCLUDE_DIR} ${ROCSPARSE_BUILD_DIR})