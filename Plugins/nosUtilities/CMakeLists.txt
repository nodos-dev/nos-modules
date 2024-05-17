# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "" nosUtilities_generated)
nos_add_plugin("nosUtilities" "stb;${NOS_SYS_VULKAN_TARGET_5_3};${NOS_PLUGIN_SDK_TARGET};nosUtilities_generated" "")
