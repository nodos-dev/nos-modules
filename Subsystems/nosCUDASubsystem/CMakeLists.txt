add_library(nosCudaSubsystem MODULE)
if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosCudaSubsystem" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "cpp")
endif()
add_folder_as_subsystem_project("nosCudaSubsystem" "" "${CMAKE_CURRENT_SOURCE_DIR}/Include")

enable_language(CUDA CXX)

find_package(CUDAToolkit REQUIRED)

message("CUDA INCLUDE DIRECTORIES: ${CUDAToolkit_INCLUDE_DIRS}")

set_target_properties(nosCudaSubsystem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nosCudaSubsystem PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(nosCudaSubsystem PRIVATE cuda)
target_link_libraries(nosCudaSubsystem PRIVATE cudart)
set(CMAKE_VERBOSE_MAKEFILE ON)

target_compile_definitions(nosCudaSubsystem PRIVATE BUILDING_NOS_VULKAN_SUBSYSTEM)
