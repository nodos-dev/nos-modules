# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

list(APPEND INCLUDE_FOLDERS ${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include)
add_folder_as_plugin_project("nosFilters" "${NOS_PLUGIN_SDK_TARGET}" ${INCLUDE_FOLDERS})