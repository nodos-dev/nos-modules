
set(PYBIND11_NOPYTHON ON CACHE BOOL "" FORCE)

add_library(python3 STATIC IMPORTED)
set_target_properties(python3 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../nosPythonDeps/3.12/libs/python3.lib)
set_target_properties(python3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../nosPythonDeps/3.12/include)


add_library(python312 STATIC IMPORTED)
set_target_properties(python312 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../nosPythonDeps/3.12/libs/python312.lib)
set_target_properties(python312 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../nosPythonDeps/3.12/include)


# You must install Python with debug binaries checked
add_subdirectory(External/pybind11 EXCLUDE_FROM_ALL)

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosPython" "cpp" "" NOS_PY_GENERATED_FILES)

nos_add_subsystem("nosPython" "${NOS_SUBSYSTEM_SDK_TARGET};python3;python312;pybind11::embed" "${CMAKE_CURRENT_SOURCE_DIR}/Include")
target_sources(nosPython PUBLIC ${NOS_PY_GENERATED_FILES})

get_targets(NOSPY_EXTERNAL_TARGETS "./External/pybind11")
group_targets("${NOSPY_EXTERNAL_TARGETS}" "External")
