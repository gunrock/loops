set(BENCHMARK_SOURCES
  work_oriented.cu
)

foreach(SOURCE IN LISTS BENCHMARK_SOURCES)
  get_filename_component(BENCHMARK_NAME ${SOURCE} NAME_WLE)
  add_executable(loops.bench.spmv.${BENCHMARK_NAME} ${SOURCE})
  target_link_libraries(loops.bench.spmv.${BENCHMARK_NAME}
    PRIVATE loops
    PRIVATE nvbench::nvbench
  )

  set_target_properties(loops.bench.spmv.${BENCHMARK_NAME}
    PROPERTIES
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
  )
  
  message(STATUS "Benchmark Added: loops.bench.spmv.${BENCHMARK_NAME}")
endforeach()