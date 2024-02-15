
if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "cpp")
endif()

add_folder_as_plugin_project("nosONNX" "" 
	"${CMAKE_CURRENT_SOURCE_DIR};${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosCUDASubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosAI/Include;${SUBSYSTEMS_SOURCE_DIR}/nosTensorSubsystem/Include")

set_target_properties(nosONNX PROPERTIES OUTPUT_NAME "nosONNX")

include_directories(${CMAKE_CURRENT_LIST_DIR}/External/include)
