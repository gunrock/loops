# begin /* Set the application name. */
set(APPLICATION_NAME range)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${APPLICATION_NAME})

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
)

target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} 
    PRIVATE loops
    PRIVATE nvToolsExt)
get_target_property(LOOPS_ARCHITECTURES loops CUDA_ARCHITECTURES)
set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${LOOPS_ARCHITECTURES}
) # XXX: Find a better way to inherit loops properties.

message(STATUS "Example Added: ${APPLICATION_NAME}")
# end /* Add CUDA executables */