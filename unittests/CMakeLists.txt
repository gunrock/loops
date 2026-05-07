# Unit tests live here. Each *.cu file becomes its own test executable
# named test.<basename>, linked against loops::loops and Catch2.
# CTest discovery is automatic via catch_discover_tests().

set(LOOPS_UNITTEST_SOURCES
  layout_contract_test.cu
)

foreach(SOURCE IN LISTS LOOPS_UNITTEST_SOURCES)
  get_filename_component(BASE_NAME ${SOURCE} NAME_WLE)
  set(TARGET_NAME test.${BASE_NAME})

  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME}
    PRIVATE
      loops::loops
      Catch2::Catch2WithMain
  )
  loops_setup_cuda_target(${TARGET_NAME})

  catch_discover_tests(${TARGET_NAME})
  message(STATUS "Test added: ${TARGET_NAME}")
endforeach()
