include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning external project: NVIDIA/cccl")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

# CCCL = the unified Thrust + CUB + libcu++ repo (the legacy thrust/thrust
# repo is archived as of CCCL 2.0). v2.7.0 supports CUDA 11.8 .. 13.x and is
# what we use when the toolkit-bundled headers are insufficient.
FetchContent_Declare(
  cccl
  GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
  GIT_TAG        v2.7.0
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(cccl)
# CCCL provides imported targets:
#   CCCL::Thrust  CCCL::CUB  CCCL::libcudacxx
