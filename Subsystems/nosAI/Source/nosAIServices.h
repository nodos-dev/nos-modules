// Copyright MediaZ AS. All Rights Reserved.

#pragma once
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
	nosResult ImportTensorFromCUDABuffer(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	nosResult ImportTensorFromVulkanResource(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorShapeInfo shapeInfo, TensorElementType elementType);
	nosResult CreateTensorFromCUDABuffer(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorCreateInfo createInfo);
	nosResult CreateTensorFromVulkanResource(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorCreateInfo createInfo);
	nosResult CreateEmptyTensor(nosTensor* tensorOut, nosTensorCreateInfo createInfo);
	nosResult InitTensor(nosTensor* tensorOut, void* MemoryAddress, nosTensorCreateInfo createInfo);
	nosResult CopyDataToTensor(nosTensor* tensorOut, void* MemoryAddress, uint64_t Size);
	nosResult SliceTensor(nosTensor* tensorIn, uint64_t* outCount, nosTensor* outTensors);
}
