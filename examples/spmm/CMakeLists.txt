set(LOOPS_SPMM_SOURCES
  thread_mapped.cu
)

foreach(SOURCE IN LISTS LOOPS_SPMM_SOURCES)
  get_filename_component(BASE_NAME ${SOURCE} NAME_WLE)
  set(TARGET_NAME loops.spmm.${BASE_NAME})

  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME} PRIVATE loops::loops)
  loops_setup_cuda_target(${TARGET_NAME})

  message(STATUS "Example added: ${TARGET_NAME}")
endforeach()
