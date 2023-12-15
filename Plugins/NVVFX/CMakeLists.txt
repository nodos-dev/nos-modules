find_library(VideoEffectsApp NAMES VideoEffectsApp.lib PATHS ${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse REQUIRED NO_DEFAULT_PATH)

add_folder_as_plugin_project("NVVFX" "VideoEffectsApp" "${CMAKE_CURRENT_SOURCE_DIR};${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include")
target_link_directories(NVVFX PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BinariesUse")