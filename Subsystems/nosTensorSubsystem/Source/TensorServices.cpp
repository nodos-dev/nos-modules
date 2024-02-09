// Copyright MediaZ AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include "TensorServices.h"
#include <cstring>
#include "CUDAKernels/TensorSlicer.ptx.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"
#include "TensorCommon.h"
#include <Nodos/Name.hpp>

extern nosVulkanSubsystem* nosVulkan = nullptr;
extern nosCUDASubsystem* nosCUDA = nullptr;

namespace nos::tensor
{

	nosResult Bind(nosTensorSubsystem* subsys) {
		nosResult returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 2, 0, (void**)&nosVulkan);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_CUDA_SUBSYSTEM_NAME), 1, 0, (void**)&nosCUDA);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		CHECK_RESULT(nosCUDA->Initialize(0));
		subsys->ImportTensorFromCUDABuffer = ImportTensorFromCUDABuffer;
		subsys->ImportTensorFromVulkanResource = ImportTensorFromVulkanResource;
		subsys->CreateTensorFromCUDABuffer = CreateTensorFromCUDABuffer;
		subsys->CreateTensorFromVulkanResource = CreateTensorFromVulkanResource;
		subsys->CreateEmptyTensor = CreateEmptyTensor;
		subsys->InitTensor = InitTensor;
		subsys->CopyDataToTensor = CopyDataToTensor;
		subsys->SliceTensor = SliceTensor;
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL ImportTensorFromCUDABuffer(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorShapeInfo shapeInfo, TensorElementType elementType)
	{
		nosTensorCreateInfo createInfo = {};
		memcpy(&createInfo.ShapeInfo, &shapeInfo, sizeof(shapeInfo));
		createInfo.ElementType = elementType;
		createInfo.Location = MEMORY_LOCATION_CUDA;

		uint64_t AllocationSize = nos::tensor::GetTensorSizeFromCreateInfo(createInfo);
		CHECK_SIZE(AllocationSize, cudaBuffer->CreateInfo.AllocationSize);

		tensorOut->MemoryInfo.Address = cudaBuffer->Address;
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL ImportTensorFromVulkanResource(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorShapeInfo shapeInfo, TensorElementType elementType)
	{
		uint64_t AllocationSize = 0;
		switch (vulkanResource->Info.Type) {
			case NOS_RESOURCE_TYPE_BUFFER:
			{
				AllocationSize = vulkanResource->Memory.Size;
				break;
			}
			case NOS_RESOURCE_TYPE_TEXTURE:
			{
				//Importing textures directly will not work due to paddings in Vulkan Texture Memory at the moment
				AllocationSize = vulkanResource->Memory.Size;
				elementType = nos::tensor::GetTensorElementTypeFromVulkanFormat(vulkanResource->Info.Texture.Format);
				//Override the element type if any
				break;
			}
			default:
			{
				CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			}
		}
		nosTensorCreateInfo createInfo = { .ElementType = elementType };
		memcpy(&createInfo.ShapeInfo, &shapeInfo, sizeof(shapeInfo)); 
		createInfo.Location = MEMORY_LOCATION_VULKAN;

		uint64_t TensorRequestedSize = nos::tensor::GetTensorSizeFromCreateInfo(createInfo);

		CHECK_SIZE(TensorRequestedSize, AllocationSize);

		tensorOut->MemoryInfo.Size = AllocationSize;
		tensorOut->MemoryInfo.Address = vulkanResource->Memory.Handle;

		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));

		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL CreateTensorFromCUDABuffer(nosTensor* tensorOut, nosCUDABufferInfo* cudaBuffer, nosTensorCreateInfo createInfo)
	{
		uint64_t AllocationSize = nos::tensor::GetTensorSizeFromCreateInfo(createInfo);
		CHECK_SIZE(AllocationSize, cudaBuffer->CreateInfo.AllocationSize);
		switch (createInfo.Location) {
		case MEMORY_LOCATION_CPU:
		{
			nosCUDABufferInfo newBuffer = {};
			CHECK_RESULT(nosCUDA->CreateBuffer(&newBuffer, AllocationSize));
			CHECK_RESULT(nosCUDA->CopyBuffers(cudaBuffer, &newBuffer));
			tensorOut->MemoryInfo.Address = newBuffer.Address;
			break;
		}
		case MEMORY_LOCATION_CUDA:
		{
			nosCUDABufferInfo newBuffer = {};
			CHECK_RESULT(nosCUDA->CreateBufferOnCUDA(&newBuffer, AllocationSize));
			CHECK_RESULT(nosCUDA->CopyBuffers(cudaBuffer, &newBuffer));
			tensorOut->MemoryInfo.Address = newBuffer.Address;
			break;
		}
		case MEMORY_LOCATION_VULKAN:
		{
			//TODO: This is an interop issue and should be handled in cuda-vulkan interop subsystem, not in tensor!
			CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		default:
		{
			CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}
		}
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL CreateTensorFromVulkanResource(nosTensor* tensorOut, nosResourceShareInfo* vulkanResource, nosTensorCreateInfo createInfo)
	{
		uint64_t AllocationSize = 0;
		TensorElementType elementType = ELEMENT_TYPE_UNDEFINED;
		switch (vulkanResource->Info.Type) {
		case NOS_RESOURCE_TYPE_BUFFER:
		{
			AllocationSize = vulkanResource->Memory.Size;
			break;
		}
		case NOS_RESOURCE_TYPE_TEXTURE:
		{
			AllocationSize = nos::tensor::GetVulkanTextureSizeLinear(vulkanResource->Info.Texture);
			//Override the element type if any
			createInfo.ElementType = nos::tensor::GetTensorElementTypeFromVulkanFormat(vulkanResource->Info.Texture.Format);
			break;
		}
		default:
		{
			CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}

		}
		uint64_t TensorRequestedSize = nos::tensor::GetTensorSizeFromCreateInfo(createInfo);

		CHECK_SIZE(TensorRequestedSize, AllocationSize);

		nosResourceShareInfo vulkanBuff = {};
		vulkanBuff.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		vulkanBuff.Info.Buffer.Size = AllocationSize;
		vulkanBuff.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
		CHECK_RESULT(nosVulkan->CreateResource(&vulkanBuff));
		nosCmd cmd;
		nosGPUEvent gpuEvent = {};
		nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &gpuEvent };
		CHECK_RESULT(nosVulkan->Begin("RGBA to tensor", &cmd));
		CHECK_RESULT(nosVulkan->Copy(cmd, vulkanResource, &vulkanBuff, 0));
		CHECK_RESULT(nosVulkan->End(&cmd, &endParams));
		CHECK_RESULT(nosVulkan->WaitGpuEvent(&gpuEvent, UINT64_MAX));
		switch (createInfo.Location) {
		case MEMORY_LOCATION_CPU:
		{
			void* data = nosVulkan->Map(&vulkanBuff);
			tensorOut->MemoryInfo.Address = reinterpret_cast<uint64_t>(data);
			break;
		}
		case MEMORY_LOCATION_CUDA:
		{
			//TODO: This is an interop issue and should be handled in cuda-vulkan interop subsystem, not in tensor!
			CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case MEMORY_LOCATION_VULKAN:
		{
			tensorOut->MemoryInfo.Address = vulkanBuff.Memory.Handle;
			break;
		}
		default:
		{
			CHECK_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}
		}
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));

		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL CreateEmptyTensor(nosTensor* tensorOut, nosTensorCreateInfo createInfo) {
		uint64_t AllocationSize = GetTensorSizeFromCreateInfo(createInfo);
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
		{
			return NOS_RESULT_FAILED;
		}
		}
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL InitTensor(nosTensor* tensorOut, void* MemoryAddress, nosTensorCreateInfo createInfo) {
		uint64_t AllocationSize = GetTensorSizeFromCreateInfo(createInfo);
		tensorOut->MemoryInfo.Address = reinterpret_cast<uint64_t>(MemoryAddress);
		tensorOut->MemoryInfo.Size = AllocationSize;
		memcpy(&tensorOut->CreateInfo, &createInfo, sizeof(createInfo));
		return NOS_RESULT_SUCCESS;
	}

	nosResult CopyDataToTensor(nosTensor* tensorOut, void* MemoryAddress, uint64_t Size)
	{
		return nosResult();
	}

	nosResult NOSAPI_CALL SliceTensor(nosTensor* tensorIn, uint64_t* outCount, nosTensor* outTensors)
	{

		return nosResult();
	}
}
