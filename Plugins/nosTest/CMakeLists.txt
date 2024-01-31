if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__")
endif()

add_folder_as_plugin_project("nosTest" "" "${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include")
