cmake_minimum_required(VERSION 3.24.2)
project("Subsystems")

function(add_folder_as_subsystem_project NAME DEPENDENCIES INCLUDE_FOLDERS)
    project(${NAME})
    message("Processing subsystem ${NAME}")

    # TODO: Set convenient paths for debug too. Or, remove it if not used.
    string(REGEX REPLACE "\\\\" "/" MZ_WORK_FOLDER "$ENV{PROGRAMDATA}/mediaz/core")
    set(MZ_WORK_FOLDER "${MZ_WORK_FOLDER}")
    set(MZ_REPO_ROOT "${MZ_WORK_FOLDER}")
    add_compile_definitions(MZ_WORK_FOLDER="${MZ_WORK_FOLDER}")
    add_compile_definitions(MZ_REPO_ROOT="${MZ_REPO_ROOT}")

    set(SOURCE_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${CMAKE_CURRENT_SOURCE_DIR}/Include")
    set(CONFIG_FOLDERS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/Config")
    foreach(folder IN LISTS SOURCE_FOLDERS)	
        message(STATUS "${PROJECT_NAME}: Scanning ${folder}")
        file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${folder} "${folder}/*.cpp" 
            "${folder}/*.cc" "${folder}/*.c" "${folder}/*.inl"
            "${folder}/*.frag" "${folder}/*.vert" "${folder}/*.glsl" "${folder}/*.comp" "${folder}/*.natvis")
        file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS ${folder} "${folder}/*.h" "${folder}/*.hpp")
        file(GLOB_RECURSE RESOURCES CONFIGURE_DEPENDS ${folder} "${folder}/*.rc")
        list(APPEND COLLECTED_SOURCES ${SOURCES})
        list(APPEND COLLECTED_HEADERS ${HEADERS})
        list(APPEND COLLECTED_RESOURCES ${RESOURCES})
    endforeach()
    foreach(CONFIG_FOLDER ${CONFIG_FOLDERS})
        file(GLOB_RECURSE CUR_CONFIG_FILES CONFIGURE_DEPENDS ${CONFIG_FOLDER} 
            "${CONFIG_FOLDER}/*.mzsubsystem" "${CONFIG_FOLDER}/*.fbs")
        list(APPEND CONFIG_FILES ${CUR_CONFIG_FILES})
    endforeach()

    add_library(${NAME} MODULE ${COLLECTED_SOURCES} ${COLLECTED_HEADERS} ${COLLECTED_RESOURCES} ${CONFIG_FILES})
    set_target_properties(${NAME} PROPERTIES
        CXX_STANDARD 20
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/Binaries"
    )

    target_include_directories(${NAME} PRIVATE ${INCLUDE_FOLDERS})

    target_link_libraries(${NAME} PRIVATE ${DEPENDENCIES})
    target_link_libraries(${NAME} PUBLIC mzSubsystemSDK)
endfunction()

# Dependencies
# ------------

# Update subsystems.json if you add dependencies, so that CD can create releases.

# MediaZ Subsystem SDK
if (NOT WITH_MEDIAZ)
    list(APPEND CMAKE_MODULE_PATH $ENV{MZ_SDK_DIR}/cmake)
    find_package(mzSubsystemSDK REQUIRED)
endif()

add_subdirectory(mzTestSubsystem)
add_subdirectory(mzVulkanSubsystem) # Subsystem prefix is only for distinguishing between mzVulkan repo and this.
