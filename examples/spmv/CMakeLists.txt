# begin /* Add application */
set(SOURCES
  original.cu  
  thread_mapped.cu
  group_mapped.cu
  work_oriented.cu
)

get_target_property(LOOPS_ARCHITECTURES 
  loops CUDA_ARCHITECTURES
)

foreach(SOURCE IN LISTS SOURCES)
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(${TEST_NAME} ${SOURCE})
  target_link_libraries(${TEST_NAME} PRIVATE loops)
  set_target_properties(${TEST_NAME} 
    PROPERTIES 
      CUDA_ARCHITECTURES ${LOOPS_ARCHITECTURES}
  )
  message(STATUS "Example Added: spmv/${TEST_NAME}")
endforeach()
# end /* Add application */