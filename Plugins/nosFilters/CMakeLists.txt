# Copyright MediaZ AS. All Rights Reserved.

list(APPEND INCLUDE_FOLDERS ${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include)
add_folder_as_plugin_project("nosFilters" "" ${INCLUDE_FOLDERS})