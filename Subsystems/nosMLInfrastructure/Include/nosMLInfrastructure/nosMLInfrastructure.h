#pragma once
#ifndef NOS_ML_INFRASTRUCTURE_H_INCLUDED
#define NOS_ML_INFRASTRUCTURE_H_INCLUDED

#include "Nodos/Types.h"
//dummy...
#define CREATE_TENSOR_SHAPE(...)\
	{ (int[] f){__VA_ARGS__};} \
	{int size = sizeof(f)/(sizeof(f[0]));} \
	{.Dimensions = f, .Size = size}; \

#pragma region Type Definitions

#pragma endregion


#pragma region Enums
typedef enum TensorMemoryLocation {
	MEMORY_LOCATION_CPU = 0x0000,
	MEMORY_LOCATION_CUDA = 0x001,
	MEMORY_LOCATION_VULKAN = 0x0002,
	//MEMORY_LOCATION_DIRECTX,
}TensorMemoryLocation;

typedef enum TensorElementType {
	ELEMENT_TYPE_UINT8,
	ELEMENT_TYPE_UINT16,
	ELEMENT_TYPE_UINT32,
	ELEMENT_TYPE_UINT64,
	ELEMENT_TYPE_INT8,
	ELEMENT_TYPE_INT16,
	ELEMENT_TYPE_INT32,
	ELEMENT_TYPE_INT64,
	ELEMENT_TYPE_STRING,
	ELEMENT_TYPE_FLOAT,
	ELEMENT_TYPE_DOUBLE,
	ELEMENT_TYPE_BOOL,
}TensorElementType;
#pragma endregion

#pragma region Structs
typedef struct nosTensorShapeInfo {
	int* Dimensions;
	uint64_t Size;
}nosTensorShapeInfo;

typedef struct nosTensorCreateInfo {
	TensorMemoryLocation Location;
	nosTensorShapeInfo ShapeInfo;
	TensorElementType ElementType;
}nosTensorCreateInfo;

typedef struct nosTensorMemoryInfo {
	uint64_t Address;
	uint64_t Size;
}nosTensorMemoryInfo;


typedef struct nosTensor {
	nosTensorMemoryInfo MemoryInfo;
	nosTensorCreateInfo CreateInfo;
}nosTensor;
#pragma endregion



typedef struct nosMLInfrastructure
{
	nosResult(NOSAPI_CALL* CreateEmptyTensor)(nosTensor* tensorOut, nosTensorCreateInfo createInfo);
	nosResult(NOSAPI_CALL* InitTensor)(nosTensor* tensorOut,void* MemoryAddress, nosTensorCreateInfo createInfo);
	nosResult(NOSAPI_CALL* CopyDataToTensor)(nosTensor* tensorOut,void* MemoryAddress, uint64_t Size);
	nosResult(NOSAPI_CALL* SliceTensor)(nosTensor* tensorIn,uint64_t* outCount, nosTensor* outTensors);
	nosResult(NOSAPI_CALL* CreateTensorPin)();

} nosMLInfrastructure;

extern nosMLInfrastructure* nosMLInfra;
#define NOS_ML_INFRASTRUCTURE_NAME "nos.ml.infra"
#endif //NOS_ML_INFRASTRUCTURE_H_INCLUDED