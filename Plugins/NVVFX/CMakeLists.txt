add_folder_as_plugin_project("NVVFX" "" "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/src;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosCUDASubsystem/Include")

target_link_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse")
target_include_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/External/nvvfx/include")
set(CMAKE_VERBOSE_MAKEFILE ON)
