cmake_minimum_required(VERSION 3.24.2)
project("Plugins")

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External)

function(add_folder_as_plugin_project NAME DEPENDENCIES INCLUDE_FOLDERS)
    project(${NAME})
    message("Processing plugin ${NAME}")

    # TODO: Set convenient paths for debug too. Or, remove it if not used.
    string(REGEX REPLACE "\\\\" "/" MZ_WORK_FOLDER "$ENV{PROGRAMDATA}/mediaz/core")
    set(MZ_WORK_FOLDER "${MZ_WORK_FOLDER}")
    set(MZ_REPO_ROOT "${MZ_WORK_FOLDER}")
    add_compile_definitions(MZ_WORK_FOLDER="${MZ_WORK_FOLDER}")
    add_compile_definitions(MZ_REPO_ROOT="${MZ_REPO_ROOT}")

    set(SOURCE_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/Source")
    set(CONFIG_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/Config")

    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${SOURCE_FOLDER}
        "${SOURCE_FOLDER}/*.cpp" "${SOURCE_FOLDER}/*.inl" "${SOURCE_FOLDER}/*.glsl"
        "${SOURCE_FOLDER}/*.comp" "${SOURCE_FOLDER}/*.frag" "${SOURCE_FOLDER}/*.vert")
    list(APPEND CONFIG_FILES)
    foreach(CONFIG_FOLDER ${CONFIG_FOLDERS})
        file(GLOB_RECURSE CUR_CONFIG_FILES CONFIGURE_DEPENDS ${CONFIG_FOLDER} 
            "${CONFIG_FOLDER}/*.mzcfg" "${CONFIG_FOLDER}/*.mzdef" "${CONFIG_FOLDER}/*.fbs")
        list(APPEND CONFIG_FILES ${CUR_CONFIG_FILES})
    endforeach()
    file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${SOURCE_FOLDER} "${SOURCE_FOLDER}/*.h" "${SOURCE_FOLDER}/*.hpp")
    file(GLOB_RECURSE RESOURCES CONFIGURE_DEPENDS ${SOURCE_FOLDER} "${SOURCE_FOLDER}/*.rc")

    set(SHADER_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/Shaders")
    list(APPEND SHADERS)
    
    foreach(SHADER_FOLDER ${SHADER_FOLDERS})
        file(GLOB_RECURSE CUR_SHADERS CONFIGURE_DEPENDS ${SHADER_FOLDER} 
            "${SHADER_FOLDER}/*.glsl" "${SHADER_FOLDER}/*.comp" "${SHADER_FOLDER}/*.frag" "${SHADER_FOLDER}/*.vert")
        list(APPEND SHADERS ${CUR_SHADERS})
    endforeach()
    
    add_library(${NAME} MODULE ${SOURCES} ${SHADERS} ${HEADERS} ${RESOURCES} ${CONFIG_FILES})
    set_target_properties(${NAME} PROPERTIES
        CXX_STANDARD 20
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    )

    target_include_directories(${NAME} PRIVATE ${INCLUDE_FOLDERS})

    target_link_libraries(${NAME} PRIVATE ${DEPENDENCIES})
    target_link_libraries(${NAME} PUBLIC mzPluginSDK)
endfunction()

# Dependencies
# ------------

# Update plugins.json if you add dependencies, so that CD can create releases.

# stb
if (NOT WITH_MEDIAZ AND NOT TARGET stb)
    add_subdirectory(${EXTERNAL_DIR}/stb "${CMAKE_CURRENT_BINARY_DIR}/stb")
endif()

# MediaZ Plugin SDK
if (NOT WITH_MEDIAZ)
    list(APPEND CMAKE_MODULE_PATH $ENV{MZ_SDK_DIR}/cmake)
    find_package(mzPluginSDK REQUIRED)
endif()

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

add_subdirectory(mzMath)
add_subdirectory(mzAJA)
add_subdirectory(mzFilters)
add_subdirectory(mzUtilities)
add_subdirectory(mzTest)
add_subdirectory(mzCNN)
add_subdirectory(mzWebRTC)

