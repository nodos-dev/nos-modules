# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

set(wmf_libs ole32.lib mf.lib mfuuid.lib mfreadwrite.lib Shlwapi.lib mfplat.lib)

list(APPEND DEPENDENCIES ${NOS_PLUGIN_SDK_TARGET} ${NOS_SYS_VULKAN_TARGET} ${wmf_libs})

nos_add_plugin("nosWebcam" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")
