// Copyright MediaZ AS. All Rights Reserved.

#ifndef TENSOR_SERVICES_H_INCLUDED
#define TENSOR_SERVICES_H_INCLUDED
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#define CHECK_SIZE(size1, size2)\
	do{\
		if (size1 != size2) { \
			nosEngine.LogE("Tensor size is different from the resource.");\
			return NOS_RESULT_FAILED;\
		}\
	} while (0);\

#define CHECK_RESULT(nosRes) \
	do { \
		nosResult result = nosRes; \
		if (result != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %d.",__FILE__, __LINE__, result); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

namespace nos::tensor
{
	nosResult Bind(nosTensorSubsystem* subsys);
	nosResult ImportTensorFromCUDABuffer(nosTensorInfo* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	nosResult ImportTensorFromVulkanResource(nosTensorInfo* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	nosResult CreateTensorFromCUDABuffer(nosTensorInfo* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorCreateInfo createInfo);
	nosResult CreateTensorFromVulkanResource(nosTensorInfo* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorCreateInfo createInfo);
	nosResult CreateEmptyTensor(nosTensorInfo* tensorOut, nosTensorCreateInfo createInfo);
	nosResult InitTensor(nosTensorInfo* tensorOut, void* MemoryAddress, nosTensorCreateInfo createInfo);
	nosResult CopyDataToTensor(nosTensorInfo* tensorOut, void* MemoryAddress, uint64_t Size);
	nosResult SliceTensor(nosTensorInfo* tensorIn, uint64_t* outCount, nosTensorInfo* outTensors);
	nosResult CreateTensorPin(nosTensorInfo* tensor, nosUUID* NodeUUID, nosUUID* GeneratedPinUUID, TensorPinConfig config);
	nosResult UpdateTensorPin(nosTensorInfo* tensor, nosUUID* NodeUUID, nosUUID* GeneratedPinUUID, TensorPinConfig config);
	nosResult RemoveTensorPin(nosTensorInfo* tensor, nosUUID* NodeUUID, nosUUID* GeneratedPinUUID);
	nosResult GetTensorElementTypeFromVulkanResource(TensorElementType* type, nosResourceShareInfo* vulkanResource);


	namespace type
	{
		nosBool CanConnectPins(const nosBuffer* srcPinData, const nosBuffer* dstPinData);
	}
}


#endif //TENSOR_SERVICES_H_INCLUDED