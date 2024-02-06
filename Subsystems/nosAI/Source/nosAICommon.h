#pragma once
#ifndef AI_COMMON_H_INCLUDED
#define AI_COMMON_H_INCLUDED
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"

typedef enum ONNXExecutionMode {
	EXECUTION_MODE_SEQUENTIAL = 0,
	EXECUTION_MODE_PARALLEL = 1,
};

typedef enum ONNXRunLocation{
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

#endif //AI_COMMON_H_INCLUDED
