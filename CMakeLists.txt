cmake_minimum_required(VERSION 3.24.2)
project("Plugins")

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

# Dependencies
# ------------

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

add_subdirectory(mzMath)
add_subdirectory(mzAJA)
add_subdirectory(mzFilters)
add_subdirectory(mzUtilities)
add_subdirectory(mzCyclorama)
add_subdirectory(zdFrustum)

