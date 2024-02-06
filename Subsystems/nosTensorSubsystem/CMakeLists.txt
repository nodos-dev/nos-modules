add_library(nosTensorSubsystem MODULE)

if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosTensorSubsystem" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__")
endif()

message("Will generate PTX headers, given file: ${CMAKE_CURRENT_LIST_DIR}/Source")
execute_process(COMMAND python "${CMAKE_CURRENT_LIST_DIR}/generate_cuda_kernels.py" -folder="${CMAKE_CURRENT_LIST_DIR}/Source" WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_folder_as_subsystem_project("nosTensorSubsystem" "" "${CMAKE_CURRENT_SOURCE_DIR}/Include;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosCudaSubsystem/Include")
set(DEPENDENCIES "nosVulkanSubsystem;nosCudaSubsystem")
set(CMAKE_VERBOSE_MAKEFILE ON)
