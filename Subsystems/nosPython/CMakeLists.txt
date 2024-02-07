# You must install Python with debug binaries checked
add_subdirectory(External/pybind11 EXCLUDE_FROM_ALL)

if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosPython" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "cpp")
    # TODO: Python custom types
    # generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosPython" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__" "python")
endif()

add_folder_as_subsystem_project("nosPython" "pybind11::embed" "${CMAKE_CURRENT_SOURCE_DIR}/Include")

get_targets(NOSPY_EXTERNAL_TARGETS "./External/pybind11")
group_targets("${NOSPY_EXTERNAL_TARGETS}" "External")
