# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Types" "${CMAKE_CURRENT_SOURCE_DIR}/Source" "cpp" "${NOS_SDK_DIR}/types" nosVariableSubsystem_generated)

list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_SUBSYSTEM_SDK_TARGET})
list(APPEND MODULE_DEPENDENCIES_TARGETS nosVariableSubsystem_generated)

list(APPEND INCLUDE_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/Include")
nos_add_subsystem("nosVariableSubsystem" "${MODULE_DEPENDENCIES_TARGETS}" "${INCLUDE_FOLDERS}")
