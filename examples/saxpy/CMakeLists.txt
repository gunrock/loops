set(APPLICATION_NAME loops.saxpy)

add_executable(${APPLICATION_NAME} saxpy.cu)
target_link_libraries(${APPLICATION_NAME} PRIVATE loops::loops)
loops_setup_cuda_target(${APPLICATION_NAME})

set_target_properties(${APPLICATION_NAME} PROPERTIES OUTPUT_NAME saxpy)

message(STATUS "Example added: ${APPLICATION_NAME}")
