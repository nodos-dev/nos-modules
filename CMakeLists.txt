# Copyright MediaZ AS. All Rights Reserved.

cmake_minimum_required(VERSION 3.24.2)
project("NOSModules")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

macro(get_targets targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_targets(${targets} ${subdir})
    endforeach()
    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(subtarget ${current_targets}) 
        if(TARGET ${subtarget})
            list(APPEND ${targets} ${subtarget})
        endif()
    endforeach()
endmacro()

macro(group_targets targets folder_name)
    foreach(target ${targets})
        get_target_property(FOLD ${target} FOLDER)
        if(${FOLD} STREQUAL "FOLD-NOTFOUND")
            set(FOLD_NAME "${folder_name}")
        else()
            set(FOLD_NAME "${folder_name}/${FOLD}")
        endif()
        set_target_properties(${target} PROPERTIES FOLDER ${FOLD_NAME})
    endforeach()
endmacro()

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/GenerateFlatbuffers.cmake)
set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Subsystems/nosShaderCompiler/Binaries/tools/${CMAKE_SYSTEM_NAME})

set(SUBSYSTEMS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Subsystems)
add_subdirectory("Subsystems" "${CMAKE_CURRENT_BINARY_DIR}/Subsystems")
get_targets(SUBSYSTEM_TARGETS "${CMAKE_CURRENT_BINARY_DIR}/Subsystems")
group_targets("${SUBSYSTEM_TARGETS}" "NOS Subsystems")

add_subdirectory("Plugins" "${CMAKE_CURRENT_BINARY_DIR}/Plugins")
get_targets(PLUGIN_TARGETS "${CMAKE_CURRENT_BINARY_DIR}/Plugins")
group_targets("${PLUGIN_TARGETS}" "NOS Plugins")