# Copyright MediaZ Teknoloji A.S. All Rights Reserved.
set(PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Include")
nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${PUBLIC_INCLUDE_DIR}/nosTrack" "cpp" "${NOS_SDK_DIR}/types" nosTrack_generated)

set(MODULE_DEPENDENCIES "nos.sys.animation-1.5")
set(dep_idx 0)
foreach(module_name_version ${MODULE_DEPENDENCIES})
    # module_name_version: <MODULE_NAME>-<MODULE_VERSION>
    string(REPLACE "-" ";" module_name_version ${module_name_version})
    list(GET module_name_version 0 module_name)
    list(GET module_name_version 1 module_version)
    nos_get_module("${module_name}" "${module_version}" DEP_${dep_idx})
    list(APPEND MODULE_DEPENDENCIES_TARGETS ${DEP_${dep_idx}})
endforeach()

list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_PLUGIN_SDK_TARGET} nosTrack_generated)


nos_add_plugin("nosTrack" "${MODULE_DEPENDENCIES_TARGETS}" "${CMAKE_CURRENT_LIST_DIR}/External/asio/asio/include;${PUBLIC_INCLUDE_DIR}/nosTrack")
