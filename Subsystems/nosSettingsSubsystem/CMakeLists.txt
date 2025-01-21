set(USE_NOS_SDK_VERSION "1.3.0")
nos_find_sdk("1.3.0" NOS_PLUGIN_SDK_TARGET NOS_SUBSYSTEM_SDK_TARGET NOS_SDK_DIR)
if (NOT DEFINED NOS_SDK_DIR)
    message(FATAL_ERROR "Nodos SDK with version ${USE_NOS_SDK_VERSION} not found, please either install it or choose different version")
endif()

set(FLATC_EXECUTABLE ${NOS_SDK_DIR}/bin/flatc)
nos_generate_flatbuffers("${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosSettingsSubsystem" "cpp" "${NOS_SDK_DIR}/types" nosSettingsSubsystem_generated)
list(APPEND MODULE_DEPENDENCIES_TARGETS ${NOS_SUBSYSTEM_SDK_TARGET} nosSettingsSubsystem_generated)

list(APPEND INCLUDE_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/Include")
nos_add_subsystem("nosSettingsSubsystem" "${MODULE_DEPENDENCIES_TARGETS}" "${INCLUDE_FOLDERS}")

#Helpers need C++20
set_target_properties("nosSettingsSubsystem" PROPERTIES CXX_STANDARD 20)