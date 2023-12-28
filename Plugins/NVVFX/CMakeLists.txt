find_library(VideoEffectsApp NAMES VideoEffectsApp.lib PATHS ${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse REQUIRED NO_DEFAULT_PATH)


add_folder_as_plugin_project("NVVFX" "VideoEffectsApp" "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/src;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include")

find_package(CUDAToolkit)
target_include_directories(NVVFX PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

message("CUDA INCLUDE DIRECTORIES: ${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(NVVFX PRIVATE cuda)
target_link_libraries(NVVFX PRIVATE cudart)
target_link_directories(NVVFX PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/../lib/x64)
target_link_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse")

target_include_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/include")