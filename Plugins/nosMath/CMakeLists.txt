# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

set(TINYEXPR_SOURCES External/tinyexpr-cpp/tinyexpr.cpp
                     External/tinyexpr-cpp/tinyexpr.h)
add_library(tinyexpr_cpp STATIC ${TINYEXPR_SOURCES})
target_include_directories(tinyexpr_cpp PUBLIC External/tinyexpr-cpp)
nos_group_targets("tinyexpr_cpp" "External")

set(DEPENDENCIES ${NOS_PLUGIN_SDK_TARGET} tinyexpr_cpp)

nos_add_plugin("nosMath" "${DEPENDENCIES}" "")
