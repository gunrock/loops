# begin /* Add application */
set(SOURCES
  thread_mapped.cu
)

foreach(SOURCE IN LISTS SOURCES)
  set_source_files_properties(${SOURCE} 
    PROPERTIES 
        LANGUAGE HIP
  )
  get_filename_component(TEST_NAME ${SOURCE} NAME_WLE)
  add_executable(loops.spmm.${TEST_NAME} ${SOURCE})
  target_link_libraries(loops.spmm.${TEST_NAME} PRIVATE loops)
  set_target_properties(loops.spmm.${TEST_NAME} 
    PROPERTIES 
      LANGUAGE HIP
      HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}
  )
  message(STATUS "Example Added: loops.spmm.${TEST_NAME}")
endforeach()
# end /* Add application */