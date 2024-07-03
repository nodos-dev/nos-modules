# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "" nosMediaIO_generated)
nos_add_plugin("nosMediaIO" "stb;nosMediaIO_generated;${NOS_SYS_VULKAN_TARGET_5_6};${NOS_PLUGIN_SDK_TARGET}" "")
