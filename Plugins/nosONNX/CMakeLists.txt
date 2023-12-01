
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/External/CPU)

find_library(onnxruntime.lib NAMES onnxruntime.lib PATHS ${CMAKE_CURRENT_LIST_DIR}/External/CPU/lib REQUIRED NO_DEFAULT_PATH)

link_directories(${CMAKE_CURRENT_LIST_DIR}/External/CPU/lib)

add_folder_as_plugin_project("nosONNX" 
			"onnxruntime.lib"
 "${CMAKE_CURRENT_SOURCE_DIR}")

#Always force MD (because webrtc.lib built in Release => MD )
target_compile_options(nosONNX PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/MD>)
set_target_properties(nosONNX PROPERTIES OUTPUT_NAME "nosONNX")

include_directories(${CMAKE_CURRENT_LIST_DIR}/External/CPU/include)
