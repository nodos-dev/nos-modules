# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "" nosReflect_generated)

set(MODULE_DEPENDENCIES "nos.sys.vulkan-6.0" "nos.sys.variables-1.0")
set(dep_idx 0)
foreach(module_name_version ${MODULE_DEPENDENCIES})
    # module_name_version: <MODULE_NAME>-<MODULE_VERSION>
    string(REPLACE "-" ";" module_name_version ${module_name_version})
    list(GET module_name_version 0 module_name)
    list(GET module_name_version 1 module_version)
    nos_get_module("${module_name}" "${module_version}" DEP_${dep_idx})
    list(APPEND MODULE_DEPENDENCIES_TARGETS ${DEP_${dep_idx}})
endforeach()
list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_PLUGIN_SDK_TARGET} nosReflect_generated)

set(INCLUDE_FOLDERS "")
nos_add_plugin("nosReflect" "${MODULE_DEPENDENCIES_TARGETS}" "${INCLUDE_FOLDERS}")

# Helpers need C++20
set_target_properties("nosReflect" PROPERTIES CXX_STANDARD 20)