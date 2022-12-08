# begin /* Set the application name. */
set(APPLICATION_NAME saxpy)
# end /* Set the application name. */

# begin /* Add HIP executables */
add_executable(${APPLICATION_NAME})

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
)

set_source_files_properties(${SOURCE_LIST} 
    PROPERTIES 
        LANGUAGE HIP
)

target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} 
    PRIVATE loops
    # PRIVATE nvToolsExt
)

set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}
) # XXX: Find a better way to inherit loops properties.

message(STATUS "Example Added: ${APPLICATION_NAME}")
# end /* Add HIP executables */