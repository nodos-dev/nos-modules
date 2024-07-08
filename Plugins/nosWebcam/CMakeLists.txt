# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

set(wmf_libs ole32.lib mf.lib mfuuid.lib mfreadwrite.lib Shlwapi.lib mfplat.lib)


nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "${NOS_SDK_DIR}/types" nosWebcam_generated)

list(APPEND DEPENDENCIES ${NOS_PLUGIN_SDK_TARGET} ${NOS_SYS_VULKAN_TARGET} ${wmf_libs} nosWebcam_generated)

nos_add_plugin("nosWebcam" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")

set_target_properties(nosWebcam PROPERTIES CXX_STANDARD 23)