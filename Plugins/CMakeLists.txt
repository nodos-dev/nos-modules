# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

cmake_minimum_required(VERSION 3.24.2)
project("Plugins")

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

if (NOT WITH_NODOS_WORKSPACE)
    include(${SCRIPTS_DIR}/CMake/Plugins.cmake)
endif()

# Dependencies
# ------------

# Update plugins.json if you add dependencies, so that CD can create releases.

# Common dependencies
add_subdirectory(External)

unset(PYTHON_PATH CACHE)
find_program(PYTHON_PATH "python")
if(PYTHON_PATH STREQUAL "")
    message("Python not found. Looking for python3.")
    find_program(PYTHON_PATH "python3")
    if(PYTHON_PATH STREQUAL "")
        message(FATAL_ERROR "Could not find python")
    else()
        message("Found python3 ${PYTHON_PATH}")
    endif()
else()
    message("Found python ${PYTHON_PATH}")
endif()

execute_process(COMMAND python  "${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.py" RESULT_VARIABLE COMPILE_SHADERS_RESULT)

if (NOT ${COMPILE_SHADERS_RESULT} EQUAL "0")
    message(FATAL_ERROR "Failed to compile shaders. Process returned ${COMPILE_SHADERS_RESULT}.")
endif()

add_subdirectory(nosMath)
add_subdirectory(nosAJA)
add_subdirectory(nosFilters)
add_subdirectory(nosUtilities)
add_subdirectory(nosTest)
add_subdirectory(nosWebRTC)
add_subdirectory(nosMediaIO)
add_subdirectory(nosWebcam)

get_targets(PLUGINS_COMMON_EXTERNAL_TARGETS "./External")
group_targets("${PLUGINS_COMMON_EXTERNAL_TARGETS}" "External")
