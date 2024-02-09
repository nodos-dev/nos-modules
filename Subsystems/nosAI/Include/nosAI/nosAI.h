#pragma once
#ifndef NOS_AI_SUBSYSTEM_H_INCLUDED
#define NOS_AI_SUBSYSTEM_H_INCLUDED

#include "Nodos/Types.h"
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"


#pragma region Type Definitions
typedef void* Model;
#pragma endregion

#pragma region Enums
typedef enum ONNXExecutionMode {
	EXECUTION_MODE_SEQUENTIAL = 0,
	EXECUTION_MODE_PARALLEL = 1,
}ONNXExecutionMode;

typedef enum ONNXRunLocation {
	RUN_ONNX_ON_CPU = 0x0000,
	RUN_ONNX_ON_CUDA = 0x0001,
	RUN_ONNX_ON_TENSORRT = 0x0002,
	RUN_ONNX_ON_DIRECTML = 0x0003,
	RUN_ONNX_ON_ROCM = 0x0004,
}ONNXRunLocation;

typedef enum ONNXGraphOptimizationLevel { //See https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html for details
	OPTIMIZATION_DISABLE_ALL = 0,
	OPTIMIZATION_ENABLE_BASIC = 1,
	OPTIMIZATION_ENABLE_EXTENDED = 2,
	OPTIMIZATION_ENABLE_ALL = 99
}ONNXGraphOptimizationLevel;

typedef enum ONNXLogLevel {
	LOG_LEVEL_VERBOSE,  ///< Verbose informational messages (least severe).
	LOG_LEVEL_INFO,     ///< Informational messages.
	LOG_LEVEL_WARNING,  ///< Warning messages.
	LOG_LEVEL_ERROR,    ///< Error messages.
	LOG_LEVEL_FATAL,    ///< Fatal error messages (most severe).
}ONNXLogLevel;

typedef enum ModelParameterMemoryLocation {
	PARAMETER_MEMORY_LOCATION_CPU,
	PARAMETER_MEMORY_LOCATION_CUDA,
}ModelParameterMemoryLocation;

typedef enum ModelFormat {
	MODEL_FORMAT_ONNX = 0x0000,
	MODEL_FORMAT_PTH = 0x0001,
	MODEL_FORMAT_H5 = 0x0002,
	MODEL_FORMAT_KERAS = 0x0003
}ModelFormat;

typedef enum IOType {
	TYPE_UNKNOWN,
	TYPE_TENSOR,
	TYPE_SEQUENCE,
	TYPE_MAP,
	TYPE_OPAQUE,
	TYPE_SPARSETENSOR,
	TYPE_OPTIONAL,
}IOType;
#pragma endregion

#pragma region Structs
typedef struct ParameterMemoryInfo {
	ModelParameterMemoryLocation Location;
	int* CUDADeviceID; //If memory in CUDA, in which device? (set to NULL for non-cuda memory)
	uint64_t Size;
}ParameterMemoryInfo;

typedef struct ModelIO {
	nosTensorShapeInfo Shape;
	TensorElementType ElementType;
//	char* Name; moved to one level up because we will need it as an array of char arrays in onnx run
	IOType Type;
}ModelIO;

typedef struct ONNXModel {
	Model Model;
	ModelIO* Inputs;
	char** InputNames;
	uint32_t InputCount;
	ModelIO* Outputs;
	char** OutputNames;
	uint32_t OutputCount;
}ONNXModel;

typedef struct ONNXInferenceThreads {
	//For CPU Inference!
	uint32_t InteropThreads;
	uint32_t IntraopThreads;
} ONNXInferenceThreads;

typedef struct ONNXCUDAOptions {
	int DeviceID;
	nosCUDAStream Stream;
	bool CustomStream;
	uint64_t MemorySize; //Set to UINT64_MAX if you dont need to limit
}ONNXCUDAOptions;

typedef struct ONNXLoadConfig {
	char LogID[256];
	ONNXRunLocation RunLocation;
	ONNXGraphOptimizationLevel OptimizationLevel;
	const wchar_t* OptimizedModelSavePath;
	ONNXExecutionMode* ExecutionMode; //Optional
	ONNXInferenceThreads* ThreadInfo; //Optional
	ONNXCUDAOptions* CUDAOptions; //Optional
}ONNXLoadConfig;

#pragma endregion



typedef struct nosAISubsystem
{
	nosResult(NOSAPI_CALL* LoadONNXModel)(ONNXModel* model, const char* path, ONNXLoadConfig config);
	nosResult(NOSAPI_CALL* RunONNXModel)(ONNXModel* model);
	nosResult(NOSAPI_CALL* SetONNXModelInput)(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);
	nosResult(NOSAPI_CALL* SetONNXModelOutput)(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);
	
} nosAISubsystem;

extern nosAISubsystem* nosAI;
#define NOS_AI_SUBSYSTEM_NAME "nos.sys.ai"
#endif //NOS_AI_SUBSYSTEM_H_INCLUDED