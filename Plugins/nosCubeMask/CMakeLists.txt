
if(NOT TARGET glm)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/External/glm)
endif()

list(APPEND DEPENDENCIES glm)
list(APPEND INCLUDE_FOLDERS 
    ${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include
    ${CMAKE_CURRENT_LIST_DIR}/External/glm)
add_folder_as_plugin_project("nosCubeMask" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")
