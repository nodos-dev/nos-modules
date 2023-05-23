cmake_minimum_required(VERSION 3.24.2)
project("Plugins")

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)


# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../CMakeBuild/Debug/install/SDK/cmake")
add_subdirectory(mzMath)
add_subdirectory(mzAJA)
add_subdirectory(mzFilters)
