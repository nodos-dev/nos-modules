# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosAnimationSubsystem" "cpp" "${NOS_SDK_DIR}/types" nosAnimationSubsystem_generated)

list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_SUBSYSTEM_SDK_TARGET})
list(APPEND MODULE_DEPENDENCIES_TARGETS nosAnimationSubsystem_generated)

list(APPEND INCLUDE_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/Include")
nos_add_subsystem("nosAnimationSubsystem" "${MODULE_DEPENDENCIES_TARGETS}" "${INCLUDE_FOLDERS}")
