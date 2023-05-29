# Dependencies
# ------------
# AJA
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

list(APPEND DEPENDENCIES stb)
list(APPEND INCLUDE_FOLDERS)
add_folder_as_plugin_project("mzCyclorama" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")