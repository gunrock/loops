include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning external project: NVBench")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

# Pin NVBench to a tagged release rather than `main` for reproducibility.
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        2024.07.18
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(nvbench)
