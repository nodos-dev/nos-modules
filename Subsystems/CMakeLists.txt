# Copyright MediaZ AS. All Rights Reserved.

cmake_minimum_required(VERSION 3.24.2)
project("Subsystems")

function(add_folder_as_subsystem_project NAME DEPENDENCIES INCLUDE_FOLDERS)
    project(${NAME})
    message("Processing subsystem ${NAME}")

    # TODO: Set convenient paths for debug too. Or, remove it if not used.
    string(REGEX REPLACE "\\\\" "/" NOS_WORK_FOLDER "$ENV{PROGRAMDATA}/Nodos/core")
    set(NOS_WORK_FOLDER "${NOS_WORK_FOLDER}")
    set(NOS_REPO_ROOT "${NOS_WORK_FOLDER}")
    add_compile_definitions(NOS_WORK_FOLDER="${NOS_WORK_FOLDER}")
    add_compile_definitions(NOS_REPO_ROOT="${NOS_REPO_ROOT}")

    set(SOURCE_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/Include")
    set(CONFIG_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/Config")
    foreach(folder IN LISTS SOURCE_FOLDERS)	
        message(STATUS "${PROJECT_NAME}: Scanning ${folder}")
        file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${folder} "${folder}/*.cpp" 
            "${folder}/*.cc" "${folder}/*.c" "${folder}/*.inl"
            "${folder}/*.frag" "${folder}/*.vert" "${folder}/*.glsl" "${folder}/*.comp" "${folder}/*.dat" "${folder}/*.natvis" "${folder}/*.py")
        file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${folder} "${folder}/*.h" "${folder}/*.hpp")
        file(GLOB_RECURSE RESOURCES CONFIGURE_DEPENDS ${folder} "${folder}/*.rc")
        list(APPEND COLLECTED_SOURCES ${SOURCES})
        list(APPEND COLLECTED_HEADERS ${HEADERS})
        list(APPEND COLLECTED_RESOURCES ${RESOURCES})
    endforeach()
    foreach(CONFIG_FOLDER ${CONFIG_FOLDERS})
        file(GLOB_RECURSE CUR_CONFIG_FILES CONFIGURE_DEPENDS ${CONFIG_FOLDER} 
            "${CONFIG_FOLDER}/*.nossys" "${CONFIG_FOLDER}/*.fbs" "${CONFIG_FOLDER}/Defaults.json")
        list(APPEND CONFIG_FILES ${CUR_CONFIG_FILES})
    endforeach()

    if(TARGET ${NAME})
        target_sources(${NAME} PRIVATE ${COLLECTED_SOURCES} ${COLLECTED_HEADERS} ${COLLECTED_RESOURCES} ${CONFIG_FILES})
    else()
        add_library(${NAME} MODULE ${COLLECTED_SOURCES} ${COLLECTED_HEADERS} ${COLLECTED_RESOURCES} ${CONFIG_FILES})
    endif()
    
    set_target_properties(${NAME} PROPERTIES
        CXX_STANDARD 20
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    )

	foreach(source IN LISTS COLLECTED_SOURCES)
        get_filename_component(source_path "${source}" PATH)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" source_path_compact "${source_path}")
        string(REPLACE "/" "\\" source_path_msvc "${source_path_compact}")
        source_group("${source_path_msvc}" FILES "${source}")
    endforeach()

    foreach(header IN LISTS COLLECTED_HEADERS)
        get_filename_component(header_path "${header}" PATH)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" header_path_compact "${header_path}")
        string(REPLACE "/" "\\" header_path_msvc "${header_path_compact}")
        source_group("${header_path_msvc}" FILES "${header}")
    endforeach()

	foreach(resource IN LISTS COLLECTED_RESOURCES)
        get_filename_component(resource_path "${resource}" PATH)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" resource_path_compact "${resource_path}")
        string(REPLACE "/" "\\" header_path_msvc "${resource_path_compact}")
        source_group("${resource_path_msvc}" FILES "${resource}")
    endforeach()

    target_include_directories(${NAME} PRIVATE ${INCLUDE_FOLDERS} ${SOURCE_FOLDERS})

    target_link_libraries(${NAME} PRIVATE ${DEPENDENCIES})
    target_link_libraries(${NAME} PUBLIC nosSubsystemSDK)

endfunction()

# Dependencies
# ------------

# Update subsystems.json if you add dependencies, so that CD can create releases.

# Nodos Subsystem SDK
if (NOT WITH_NODOS)
	cmake_path(SET NODOS_SDK_DIR_PATH $ENV{NODOS_SDK_DIR})
	list(APPEND CMAKE_MODULE_PATH ${NODOS_SDK_DIR_PATH}/cmake)
    find_package(nosSubsystemSDK)
endif()

add_subdirectory(nosVulkanSubsystem) # Subsystem prefix is only for distinguishing between nosVulkan repo and this.
add_subdirectory(nosTestSubsystem)
add_subdirectory(nosPython)
add_subdirectory(nosCUDASubsystem)
add_subdirectory(nosTensorSubsystem)
add_subdirectory(nosAI)
