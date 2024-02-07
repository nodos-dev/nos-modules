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
};

typedef enum ONNXRunLocation {
	RUN_ON_CPU = 0x0000,
	RUN_ON_CUDA = 0x0001,
	RUN_ON_TENSORRT = 0x0002,
	RUN_ON_DIRECTML = 0x0003,
	RUN_ON_ROCM = 0x0004,
}ONNXRunLocation;

typedef enum ONNXGraphOptimizationLevel { //See https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html for details
	DISABLE_ALL = 0,
	ENABLE_BASIC = 1,
	ENABLE_EXTENDED = 2,
	ENABLE_ALL = 99
} ONNXGraphOptimizationLevel;

typedef enum ONNXLogLevel {
	LOG_LEVEL_VERBOSE,  ///< Verbose informational messages (least severe).
	LOG_LEVEL_INFO,     ///< Informational messages.
	LOG_LEVEL_WARNING,  ///< Warning messages.
	LOG_LEVEL_ERROR,    ///< Error messages.
	LOG_LEVEL_FATAL,    ///< Fatal error messages (most severe).
}ONNXLogLevel;

typedef enum ModelLocation {
	LOCATION_CPU = 0x0000,
	LOCATION_CUDA = 0x0001,
	LOCATION_TENSORRT = 0X0002,
};

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
typedef struct ModelIO {
	nosTensorShapeInfo Shape;
	TensorElementType ElementType;
	char* Name;
	IOType Type;
}ModelIO;

typedef struct AIModel {
	Model Model;
	ModelFormat Format;
	ModelIO* Inputs;
	uint32_t InputCount;
	ModelIO* Outputs;
	uint32_t OutputCount;
}AIModel;

typedef struct InferenceThreads {
	//For CPU Inference!
	uint32_t InteropThreads;
	uint32_t IntraopThreads;
} InferenceThreads;

typedef struct ONNXCUDAOptions {
	int DeviceID;
	nosCUDAStream Stream;
	bool CustomStream;
	uint64_t MemorySize; //Set to UINT64_MAX if you dont need to limit
}ONNXCUDAOptions;

typedef struct ONNXLoadConfig {
	const char* LogID;
	ONNXRunLocation RunLocation;
	ONNXGraphOptimizationLevel OptimizationLevel;
	ONNXExecutionMode* ExecutionMode; //Optional
	InferenceThreads* ThreadInfo; //Optional
	ONNXCUDAOptions* CUDAOptions; //Optional
};

#pragma endregion



typedef struct nosAISubsystem
{
	nosResult(NOSAPI_CALL* LoadModel)(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	

} nosAISubsystem;

extern nosAISubsystem* nosAI;
#define NOS_AI_SUBSYSTEM_NAME "nos.sys.ai"
#endif //NOS_AI_SUBSYSTEM_H_INCLUDED