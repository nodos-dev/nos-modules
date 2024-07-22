set(MODULE_DEPENDENCIES "nos.py-0.1")
list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_PLUGIN_SDK_TARGET})
set(dep_idx 0)
foreach(module_name_version ${MODULE_DEPENDENCIES})
    # module_name_version: <MODULE_NAME>-<MODULE_VERSION>
    string(REPLACE "-" ";" module_name_version ${module_name_version})
    list(GET module_name_version 0 module_name)
    list(GET module_name_version 1 module_version)
    nos_get_module("${module_name}" "${module_version}" DEP_${dep_idx})
    list(APPEND MODULE_DEPENDENCIES_TARGETS ${DEP_${dep_idx}})
endforeach()

set(INCLUDE_FOLDERS "")
nos_add_plugin("nosPyUtils" "${MODULE_DEPENDENCIES_TARGETS}" "${INCLUDE_FOLDERS}")

# Collect .py files
file(GLOB_RECURSE PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.py")
target_sources("nosPyUtils" PRIVATE ${PY_FILES})
