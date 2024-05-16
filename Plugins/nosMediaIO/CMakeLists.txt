# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "" NOS_MEDIAIO_GENERATED)
nos_add_plugin("nosMediaIO" "stb;${NOS_SYS_VULKAN_TARGET_5_3};${NOS_PLUGIN_SDK_TARGET}" "")
target_sources(nosMediaIO PUBLIC ${NOS_MEDIAIO_GENERATED})