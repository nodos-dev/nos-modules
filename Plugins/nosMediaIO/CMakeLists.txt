# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "cpp" "" NOS_MEDIAIO_GENERATED)
nos_add_plugin("nosMediaIO" "stb;${NOS_PLUGIN_SDK_TARGET};${NOS_SYS_VULKAN_TARGET}" "")
target_sources(nosMediaIO PUBLIC ${NOS_MEDIAIO_GENERATED})