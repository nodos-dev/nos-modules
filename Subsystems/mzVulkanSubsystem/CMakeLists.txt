set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

add_subdirectory(${EXTERNAL_DIR}/mzVulkan)

add_folder_as_subsystem_project("mzVulkanSubsysem" "mzVulkan;stb" "${CMAKE_CURRENT_SOURCE_DIR}/Include")
