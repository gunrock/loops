# Unit tests live here. Every test_*.cu file becomes its own test executable
# named test.<basename>, linked against loops::loops and Catch2 (with main).
# CTest discovery is automatic via catch_discover_tests(), so each TEST_CASE
# inside a file shows up as an individually runnable ctest.
#
# Test layout (kept hermetic; no dataset I/O, no shared global state):
#
#   test_helpers.hxx              - synthetic-matrix factories, predicates
#   test_layout_contract.hxx      - shared invariant checkers (per-layout)
#   test_spmv_battery.hxx         - shared SpMV test-matrix battery
#   test_spmv_runner.hxx          - host->device->kernel->host plumbing
#
#   test_layout_*.cu              - layout contract conformance, 1 file/layout
#   test_container_*.cu           - ctors / conversions / cross-space copies
#   test_format_round_trip.cu     - dense-equivalence across format conversions
#   test_spmv_*.cu                - end-to-end correctness, 1 file per format
#   test_schedule_coverage.cu     - direct visit-count check on thread_mapped
#   test_util_*.cu                - math + range utilities

file(GLOB LOOPS_UNITTEST_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS
  test_*.cu
)

list(SORT LOOPS_UNITTEST_SOURCES)

foreach(SOURCE IN LISTS LOOPS_UNITTEST_SOURCES)
  get_filename_component(BASE_NAME ${SOURCE} NAME_WLE)
  set(TARGET_NAME test.${BASE_NAME})

  add_executable(${TARGET_NAME} ${SOURCE})
  target_include_directories(${TARGET_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${TARGET_NAME}
    PRIVATE
      loops::loops
      Catch2::Catch2WithMain
  )
  loops_setup_cuda_target(${TARGET_NAME})

  catch_discover_tests(${TARGET_NAME})
  message(STATUS "Test added: ${TARGET_NAME}")
endforeach()
