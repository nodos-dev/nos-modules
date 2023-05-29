
# Dependencies
# ------------
# boost math
add_library(boost_math INTERFACE)
target_include_directories(boost_math INTERFACE ${EXTERNAL_DIR})

# nlohmann
if (NOT WITH_MEDIAZ)
    add_library(nlohmann INTERFACE)
    target_include_directories(nlohmann INTERFACE ${EXTERNAL_DIR}/json/single_include)
endif()

list(APPEND DEPENDENCIES nlohmann boost_math)
add_folder_as_plugin_project("zdFrustum" "${DEPENDENCIES}" "")