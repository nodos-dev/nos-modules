add_folder_as_plugin_project("NVVFX" "VideoEffectsApp" "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/src;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include")
enable_language(CUDA CXX)

find_package(CUDAToolkit REQUIRED)

message("CUDA INCLUDE DIRECTORIES: ${CUDAToolkit_INCLUDE_DIRS}")

set_target_properties(NVVFX PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(NVVFX PRIVATE cuda)
target_link_libraries(NVVFX PRIVATE cudart)
target_link_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse")
target_include_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/include")
set(CMAKE_VERBOSE_MAKEFILE ON)
