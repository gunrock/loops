include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning external project: Catch2")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

# Catch2 v3 ships as a proper compiled library (no header-only mode); we use
# the bundled Catch2WithMain target so test binaries don't need to write a
# main() of their own.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)

# Provide CTest integration via Catch2's catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
