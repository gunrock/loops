set(LOOPS_SPMV_SOURCES
  original.cu
  thread_mapped.cu
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

# group_mapped leans on NVIDIA cooperative-groups (experimental
# block_tile_memory + exclusive_scan), which HIP does not provide yet, so it is
# a CUDA-only example.
if(NOT LOOPS_BACKEND STREQUAL "HIP")
  list(APPEND LOOPS_SPMV_SOURCES group_mapped.cu)
endif()

# Each SpMV example compiles for two precisions:
#   - <name>.f32  (LOOPS_VALUE_T=float)
#   - <name>.f64  (LOOPS_VALUE_T=double)
# Sources read `using type_t = LOOPS_VALUE_T;`, so we just feed a different
# preprocessor define per target. Adding e.g. half-precision is one extra
# entry below.
set(LOOPS_SPMV_PRECISIONS f32 f64)

foreach(SOURCE IN LISTS LOOPS_SPMV_SOURCES)
  get_filename_component(BASE_NAME ${SOURCE} NAME_WLE)
  foreach(PRECISION IN LISTS LOOPS_SPMV_PRECISIONS)
    if(PRECISION STREQUAL f32)
      set(VALUE_T float)
    elseif(PRECISION STREQUAL f64)
      set(VALUE_T double)
    else()
      message(FATAL_ERROR "Unknown SpMV precision '${PRECISION}'")
    endif()

    set(TARGET_NAME loops.spmv.${BASE_NAME}.${PRECISION})
    add_executable(${TARGET_NAME} ${SOURCE})
    target_link_libraries(${TARGET_NAME} PRIVATE loops::loops)
    target_compile_definitions(${TARGET_NAME} PRIVATE LOOPS_VALUE_T=${VALUE_T})
    loops_setup_cuda_target(${TARGET_NAME})

    message(STATUS "Example added: ${TARGET_NAME}")
  endforeach()
endforeach()
