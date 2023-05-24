cmake_minimum_required(VERSION 3.24.2)
project("mzBasic")

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

# stb
if (NOT WITH_MEDIAZ)
    add_subdirectory(${EXTERNAL_DIR}/stb "${CMAKE_CURRENT_BINARY_DIR}/stb")
endif()

# asio
add_library(asio INTERFACE)
target_include_directories(asio INTERFACE ${EXTERNAL_DIR}/asio/asio/include)

# MediaZ Plugin SDK
if (NOT WITH_MEDIAZ)
    list(APPEND CMAKE_MODULE_PATH $ENV{MZ_SDK_DIR}/cmake)
    find_package(mzPluginSDK REQUIRED)
endif()

# Plugin
# ------
set(CMAKE_CXX_STANDARD 20)

string(REGEX REPLACE "\\\\" "/" MZ_WORK_FOLDER "$ENV{PROGRAMDATA}/mediaz/core")
set(MZ_WORK_FOLDER "${MZ_WORK_FOLDER}")
set(MZ_REPO_ROOT "${MZ_WORK_FOLDER}")
add_compile_definitions(MZ_WORK_FOLDER="${MZ_WORK_FOLDER}")
add_compile_definitions(MZ_REPO_ROOT="${MZ_REPO_ROOT}")

set(SOURCE_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/Source")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${SOURCE_FOLDER}
    "${SOURCE_FOLDER}/*.cpp" "${SOURCE_FOLDER}/*.inl" "${SOURCE_FOLDER}/*.glsl"
    "${SOURCE_FOLDER}/*.comp" "${SOURCE_FOLDER}/*.frag" "${SOURCE_FOLDER}/*.vert")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${SOURCE_FOLDER} "${SOURCE_FOLDER}/*.h" "${SOURCE_FOLDER}/*.hpp")
file(GLOB_RECURSE RESOURCES CONFIGURE_DEPENDS ${SOURCE_FOLDER} "${SOURCE_FOLDER}/*.rc")

add_library(${PROJECT_NAME} MODULE ${SOURCES} ${HEADERS} ${RESOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

target_include_directories(${PROJECT_NAME} PUBLIC $ENV{MZ_SDK_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${EXTERNAL_DIR}
    ${EXTERNAL_DIR}/ntv2
    ${EXTERNAL_DIR}/ntv2/ajalibraries/ajantv2/src/win
    ${EXTERNAL_DIR}/ntv2/ajalibraries/ajantv2/includes
    ${SOURCE_FOLDER})

target_link_libraries(${PROJECT_NAME} PRIVATE ajantv2 stb asio)
target_link_libraries(${PROJECT_NAME} PUBLIC mzPluginSDK)
