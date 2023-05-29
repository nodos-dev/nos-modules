list(APPEND DEPENDENCIES asio)
list(APPEND INCLUDE_FOLDERS)
add_folder_as_plugin_project("mzTrack" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")
