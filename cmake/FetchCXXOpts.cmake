include(FetchContent)
set(FETCHCONTENT_QUIET ON)

message(STATUS "Cloning external project: cxxopts")
get_filename_component(FC_BASE "../externals"
                REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${FC_BASE})

# Prefer a system-installed cxxopts if available; only fetch as a fallback.
find_package(cxxopts QUIET CONFIG)
if(cxxopts_FOUND)
  message(STATUS "Using system cxxopts ${cxxopts_VERSION}")
  get_target_property(CXXOPTS_INCLUDE_DIR cxxopts::cxxopts
                      INTERFACE_INCLUDE_DIRECTORIES)
else()
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.2.1
    GIT_SHALLOW    TRUE
  )
  FetchContent_GetProperties(cxxopts)
  if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
  endif()
  set(CXXOPTS_INCLUDE_DIR "${cxxopts_SOURCE_DIR}/include")
endif()
