
add_library(nosAI MODULE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/External/CPU)
find_library(onnxruntime.lib NAMES onnxruntime.lib PATHS ${CMAKE_CURRENT_LIST_DIR}/External/ONNXRT/CPU/lib REQUIRED NO_DEFAULT_PATH)
find_library(onnxruntime_providers_cuda.lib NAMES onnxruntime_providers_cuda.lib PATHS ${CMAKE_CURRENT_LIST_DIR}/External/ONNXRT/GPU/lib REQUIRED NO_DEFAULT_PATH)
find_library(onnxruntime_providers_tensorrt.lib NAMES onnxruntime_providers_tensorrt.lib PATHS ${CMAKE_CURRENT_LIST_DIR}/External/ONNXRT/GPU/lib REQUIRED NO_DEFAULT_PATH)

link_directories(${CMAKE_CURRENT_LIST_DIR}/External/ONNXRT/CPU/lib)
link_directories(${CMAKE_CURRENT_LIST_DIR}/External/ONNXRT/GPU/lib)

if (GENERATE_FLATBUFFERS_FILES)
    generate_flatbuffers("${TOOLS_DIR}/flatc" "${CMAKE_CURRENT_SOURCE_DIR}/Config" "${CMAKE_CURRENT_SOURCE_DIR}/Include/nosTensorSubsystem" "${CMAKE_CURRENT_SOURCE_DIR}/__temp__")
endif()

message("Will generate PTX headers, given file: ${CMAKE_CURRENT_LIST_DIR}/Source")
execute_process(COMMAND python "${CMAKE_CURRENT_LIST_DIR}/generate_cuda_kernels.py" -folder="${CMAKE_CURRENT_LIST_DIR}/Source" WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

add_folder_as_subsystem_project("nosAI" "" "${CMAKE_CURRENT_SOURCE_DIR}/Include;${SUBSYSTEMS_SOURCE_DIR}/nosVulkanSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosCudaSubsystem/Include;${SUBSYSTEMS_SOURCE_DIR}/nosTensorSubsystem/Include;${CMAKE_CURRENT_SOURCE_DIR}/External/ONNXRT/include")
set(DEPENDENCIES "nosVulkanSubsystem;nosCudaSubsystem;nosTensorSubsystem")
set(CMAKE_VERBOSE_MAKEFILE ON)