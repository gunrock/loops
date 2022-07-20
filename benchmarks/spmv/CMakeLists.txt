set(BENCHMARK_SOURCES
  work_oriented.cu
)

foreach(SOURCE IN LISTS BENCHMARK_SOURCES)
  get_filename_component(BENCHMARK_NAME ${SOURCE} NAME_WLE)
  string(PREPEND BENCHMARK_NAME "loops.bench.spmv.")
  add_executable(${BENCHMARK_NAME} ${SOURCE})
  target_link_libraries(${BENCHMARK_NAME}
    PRIVATE loops
    PRIVATE nvbench::main
    PRIVATE nvbench::nvbench
  )

  set_target_properties(${BENCHMARK_NAME}
    PROPERTIES
      CXX_STANDARD 17
      CUDA_STANDARD 17
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  
  message(STATUS "Benchmark Added: ${BENCHMARK_NAME}")
endforeach()