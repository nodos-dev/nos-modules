#pragma once
#ifndef NOS_TENSOR_H_INCLUDED
#define NOS_TENSOR_H_INCLUDED

#include "Nodos/Types.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"

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

//Compatibility with ONNXRT is crucial for ONNX Model Running!
typedef enum TensorElementType {
	ELEMENT_TYPE_UNDEFINED,
	ELEMENT_TYPE_FLOAT,
	ELEMENT_TYPE_UINT8,
	ELEMENT_TYPE_INT8,  
	ELEMENT_TYPE_UINT16,
	ELEMENT_TYPE_INT16,
	ELEMENT_TYPE_INT32,
	ELEMENT_TYPE_INT64,
	ELEMENT_TYPE_STRING,
	ELEMENT_TYPE_BOOL,
	ELEMENT_TYPE_FLOAT16,
	ELEMENT_TYPE_DOUBLE,
	ELEMENT_TYPE_UINT32,
	ELEMENT_TYPE_UINT64,
	ELEMENT_TYPE_COMPLEX64,
	ELEMENT_TYPE_COMPLEX128,
	ELEMENT_TYPE_BFLOAT16,
	ELEMENT_TYPE_FLOAT8E4M3FN,
	ELEMENT_TYPE_FLOAT8E4M3FNUZ,
	ELEMENT_TYPE_FLOAT8E5M2,
	ELEMENT_TYPE_FLOAT8E5M2FNUZ,
	MIN = ELEMENT_TYPE_UNDEFINED,
	MAX = ELEMENT_TYPE_FLOAT8E5M2FNUZ
}TensorElementType;
#pragma endregion

#pragma region Structs
typedef struct nosTensorShapeInfo {
	int64_t* Dimensions;
	uint64_t DimensionCount;
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



typedef struct nosTensorSubsystem
{
	//Interop
	//Imports the CUDA Buffer memory as tensor, no copy performed since they point to same memory
	nosResult(NOSAPI_CALL* ImportTensorFromCUDABuffer)(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	//Imports the Vulkan Resource as tensor, no copy performed since they point to same memory. If the resource is texture, tensorElementType will be deduced from the format and given parameter will be overridden.
	//One should note that importing texture resources may not be successfull because of paddings in Texture Memory since tensors expect linear memory by nature
	nosResult(NOSAPI_CALL* ImportTensorFromVulkanResource)(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	
	//Will perform copy from the data of CUDA Buffer to newly created nosTensor
	nosResult(NOSAPI_CALL* CreateTensorFromCUDABuffer)(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorCreateInfo createInfo); 
	//Will perform copy from the data of Vulkan Resource to newly created nosTensor. If the resource is texture, createInfo.ElementType will be deduced from the format and given parameter will be overridden.
	nosResult(NOSAPI_CALL* CreateTensorFromVulkanResource)(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorCreateInfo createInfo); 

	nosResult(NOSAPI_CALL* CreateEmptyTensor)(nosTensor* tensorOut, nosTensorCreateInfo createInfo);
	nosResult(NOSAPI_CALL* InitTensor)(nosTensor* tensorOut,void* MemoryAddress, nosTensorCreateInfo createInfo);
	nosResult(NOSAPI_CALL* CopyDataToTensor)(nosTensor* tensorOut,void* MemoryAddress, uint64_t Size);
	nosResult(NOSAPI_CALL* SliceTensor)(nosTensor* tensorIn,uint64_t* outCount, nosTensor* outTensors);
	nosResult(NOSAPI_CALL* CreateTensorPin)();

} nosTensorSubsystem;

extern nosTensorSubsystem* nosTensorSubsys;
#define NOS_TENSOR_NAME "nos.sys.tensor"
#endif //NOS_TENSOR_H_INCLUDED