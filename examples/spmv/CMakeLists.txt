set(LOOPS_SPMV_SOURCES
  original.cu
  thread_mapped.cu
  group_mapped.cu
  work_oriented.cu
  merge_path.cu
  ell_thread_mapped.cu
  ell_merge_path.cu
  custom_layout.cu
  flat_partitioned.cu
  coo_thread_mapped.cu
  csc_thread_mapped.cu
  bcsr_thread_mapped.cu
  dia_thread_mapped.cu
)

foreach(SOURCE IN LISTS LOOPS_SPMV_SOURCES)
  get_filename_component(BASE_NAME ${SOURCE} NAME_WLE)
  set(TARGET_NAME loops.spmv.${BASE_NAME})

  add_executable(${TARGET_NAME} ${SOURCE})
  target_link_libraries(${TARGET_NAME} PRIVATE loops::loops)
  loops_setup_cuda_target(${TARGET_NAME})

  message(STATUS "Example added: ${TARGET_NAME}")
endforeach()
