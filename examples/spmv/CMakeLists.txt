# begin /* Add application */
set(SOURCES
  original.cu  
  thread_mapped.cu
  group_mapped.cu
  work_oriented.cu
)

foreach(SOURCE IN LISTS SOURCES)
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(loops.spmv.${TEST_NAME} ${SOURCE})
  target_link_libraries(loops.spmv.${TEST_NAME} PRIVATE loops)
  set_target_properties(loops.spmv.${TEST_NAME} 
    PROPERTIES 
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  message(STATUS "Example Added: loops.spmv.${TEST_NAME}")
endforeach()
# end /* Add application */