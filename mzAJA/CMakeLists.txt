# Dependencies
# ------------
# AJA
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# TODO: Make External/ajalibraries a submodule (https://github.com/aja-video/ntv2)?
list(APPEND CMAKE_MODULE_PATH ${EXTERNAL_DIR}/ntv2/cmake)

# Caution: AJA wants C++11 standard. Change it back to our's after processing AJA
include(AJAHelpers)
include(AJABuildOptions)
include(AJACommonDefines)
include(AJACommonFlags)
include(AJACommonPaths)
include(AJACommonVars)
include(AJAVersionConfig)
# CMake includes
include(GNUInstallDirs)
add_subdirectory(${EXTERNAL_DIR}/ntv2/ajalibraries "${CMAKE_CURRENT_BINARY_DIR}/ajalibraries")

list(APPEND DEPENDENCIES stb ajantv2)
list(APPEND INCLUDE_FOLDERS ${EXTERNAL_DIR}
    ${EXTERNAL_DIR}/ntv2
    ${EXTERNAL_DIR}/ntv2/ajalibraries/ajantv2/src/win
    ${EXTERNAL_DIR}/ntv2/ajalibraries/ajantv2/includes)
add_folder_as_plugin_project("mzAJA" "${DEPENDENCIES}" "${INCLUDE_FOLDERS}")
