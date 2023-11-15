# begin /* Add application */
set(SOURCES
  thread_mapped.cu
)

foreach(SOURCE IN LISTS SOURCES)
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(loops.spgemm.${TEST_NAME} ${SOURCE})
  target_link_libraries(loops.spgemm.${TEST_NAME} PRIVATE loops)
  set_target_properties(loops.spgemm.${TEST_NAME} 
    PROPERTIES 
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  message(STATUS "Example Added: loops.spgemm.${TEST_NAME}")
endforeach()
# end /* Add application */