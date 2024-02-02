add_library(nosMLInfrastructure MODULE)
if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosMLInfrastructure" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__")
endif()
add_folder_as_subsystem_project("nosMLInfrastructure" "" "${CMAKE_CURRENT_SOURCE_DIR}/Include;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosCudaSubsystem/Include")
set(DEPENDENCIES "nosVulkanSubsystem;nosCudaSubsystem")

set(CMAKE_VERBOSE_MAKEFILE ON)