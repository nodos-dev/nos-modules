# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "cpp")
endif()

if(NOT TARGET glfw)
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/External/glfw)	
	message("External: ${CMAKE_CURRENT_SOURCE_DIR}/External/")
	get_targets(NOSTEST_EXTERNAL_TARGETS "${CMAKE_CURRENT_SOURCE_DIR}/External/glfw")
	group_targets("${NOSTEST_EXTERNAL_TARGETS}" "External")
endif()

list(APPEND DEPENDENCIES glfw)
list(APPEND INCLUDE_FOLDERS
	${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include
	${CMAKE_CURRENT_SOURCE_DIR}/External/glfw)
add_folder_as_plugin_project("nosTest" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")
