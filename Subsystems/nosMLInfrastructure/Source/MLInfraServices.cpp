// Copyright MediaZ AS. All Rights Reserved.

#include <Nodos/SubsystemAPI.h>
#include "MLInfraServices.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"
#include <cstring>

namespace nos::ml::infra
{
	void Bind(nosMLInfrastructure* subsys) {

	}

	nosResult CreateEmptyTensor(nosTensor* tensorOut, nosTensorCreateInfo createInfo) {
		uint64_t AllocationSize = GetRequiredSizeFromCreateInfo(createInfo);
		switch (createInfo.Location) {
			case MEMORY_LOCATION_CPU:
			{
				void* address = malloc(AllocationSize);
				tensorOut->MemoryInfo.Address = reinterpret_cast<uint64_t>(address);
				break;
			}
			case MEMORY_LOCATION_CUDA: 
			{
				nosCUDABufferInfo CUDABuffer = {};
				nosResult resCuda = nosCUDA->CreateBufferOnCUDA(&CUDABuffer, AllocationSize);
				if (resCuda != NOS_RESULT_SUCCESS)
					return resCuda;
				tensorOut->MemoryInfo.Address = CUDABuffer.Address;
				break;
			}
			case MEMORY_LOCATION_VULKAN:
			{
				nosResourceShareInfo vulkanBuffer = {};
				vulkanBuffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
				vulkanBuffer.Info.Buffer.Size = AllocationSize;
				vulkanBuffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_DST | NOS_BUFFER_USAGE_TRANSFER_SRC);
				nosResult resVulkan = nosVulkan->CreateResource(&vulkanBuffer);
				if (resVulkan != NOS_RESULT_SUCCESS)
					return resVulkan;
				tensorOut->MemoryInfo.Address = vulkanBuffer.Memory.Handle;
				break;
			}
			default:
				return NOS_RESULT_FAILED;
		}
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult InitTensor(nosTensor* tensorOut, void* MemoryAddress, nosTensorCreateInfo createInfo) {
		uint64_t AllocationSize = GetRequiredSizeFromCreateInfo(createInfo);
		tensorOut->MemoryInfo.Address = reinterpret_cast<uint64_t>(MemoryAddress);
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult SliceTensor(nosTensor* tensorIn, uint64_t* outCount, nosTensor* outTensors)
	{

		return nosResult();
	}

	uint64_t GetRequiredSizeFromCreateInfo(nosTensorCreateInfo createInfo) {
		uint64_t totalElementCount = 1;
		for (int i = 0; i < createInfo.ShapeInfo.Size; i++) {
			totalElementCount *= createInfo.ShapeInfo.Dimensions[i];
		}
		return GetSizeOfElementType(createInfo.ElementType) * totalElementCount;
	}

	short GetSizeOfElementType(TensorElementType type) {
		switch (type) {
			case ELEMENT_TYPE_UINT8:
				return sizeof(uint8_t);
			case ELEMENT_TYPE_UINT16:
				return sizeof(uint16_t);
			case ELEMENT_TYPE_UINT32:
				return sizeof(uint32_t);
			case ELEMENT_TYPE_UINT64:
				return sizeof(uint64_t);
			case ELEMENT_TYPE_INT8:
				return sizeof(int8_t);
			case ELEMENT_TYPE_INT16:
				return sizeof(int16_t);
			case ELEMENT_TYPE_INT32:
				return sizeof(int32_t);
			case ELEMENT_TYPE_INT64:
				return sizeof(int64_t);
			case ELEMENT_TYPE_STRING:
				return sizeof(char);
			case ELEMENT_TYPE_FLOAT:
				return sizeof(float);
			case ELEMENT_TYPE_DOUBLE:
				return sizeof(double);
			case ELEMENT_TYPE_BOOL:
				return sizeof(bool);
		}
		return 0;
	}
}
