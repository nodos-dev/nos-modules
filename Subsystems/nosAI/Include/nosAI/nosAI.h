#pragma once
#ifndef NOS_AI_SUBSYSTEM_H_INCLUDED
#define NOS_AI_SUBSYSTEM_H_INCLUDED

#include "Nodos/Types.h"
#include "nosTensorSubsystem/nosTensorSubsystem.h"


#pragma region Type Definitions
typedef void* Model;
#pragma endregion

#pragma region Enums
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
#pragma endregion

#pragma region Structs
typedef struct ModelIO {
	nosTensor* Tensors;
	const char** Names;
	uint32_t Count;
}ModelIO;

typedef struct ModelInfo {
	Model Model;
	ModelFormat Format;
	ModelIO Inputs;
	ModelIO Outputs;
}ModelInfo;

#pragma endregion



typedef struct nosAISubsystem
{
	nosResult(NOSAPI_CALL* LoadModel)(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	

} nosAISubsystem;

extern nosAISubsystem* nosAI;
#define NOS_AI_SUBSYSTEM_NAME "nos.sys.ai"
#endif //NOS_AI_SUBSYSTEM_H_INCLUDED