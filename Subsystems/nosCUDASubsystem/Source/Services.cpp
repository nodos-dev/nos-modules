#include "Services.h"

// SDK
#include <Nodos/Name.hpp>
#include <nosFlatBuffersCommon.h>
#include <Nodos/SubsystemAPI.h>
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

// std
#include <future>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>

namespace nos::cudass 
{
	nosResult Initialize(int device)
	{
		//We will initialize CUDA Runtime explicitly, Driver API will also be initialized implicitly
		int cudaVersion = 0;
		cudaError res = cudaSuccess;
		
		res = cudaDriverGetVersion(&cudaVersion);
		if (cudaVersion == 0) {
			return NOS_RESULT_FAILED;
		}
		CHECK_CUDA_RT_ERROR(res);

		if (cudaVersion / 1000 >= 12) { //major version
			res = cudaSetDevice(device);
			CHECK_CUDA_RT_ERROR(res);
		}
		else {
			res = cudaFree(0); //explicit initialization pre CUDA 12.0
		}
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult GetCudaVersion(CUDAVersion* versionInfo)
	{
		int cudaVersion = 0;
		cudaError res = cudaDriverGetVersion(&cudaVersion);
		CHECK_CUDA_RT_ERROR(res);

		versionInfo->Major = cudaVersion / 1000;
		versionInfo->Minor = (cudaVersion - (cudaVersion / 1000)*1000)/10;
		return NOS_RESULT_SUCCESS;
	}
	nosResult GetDeviceCount(int* deviceCount)
	{
		cudaError res = cudaSuccess;
		res = cudaGetDeviceCount(deviceCount);
		CHECK_CUDA_RT_ERROR(res);
	}
	nosResult GetDeviceProperties(int device, nosCUDADeviceProperties* deviceProperties)
	{
		CHECK_VALID_ARGUMENT(deviceProperties);
		cudaDeviceProp deviceProp = {};
		cudaError res = cudaGetDeviceProperties(&deviceProp, device);
		CHECK_CUDA_RT_ERROR(res);
		deviceProperties->ComputeCapabilityMajor = deviceProp.major;
		deviceProperties->ComputeCapabilityMinor = deviceProp.minor;
		memcpy(deviceProp.uuid.bytes, deviceProperties->DeviceUUID, UUID_SIZE);
		memcpy(deviceProp.name, deviceProperties->Name, DEVICE_NAME_SIZE);
		return NOS_RESULT_SUCCESS;
	}
	nosResult CreateStream(nosCUDAStream* stream)
	{
		cudaStream_t cudaStream;
		cudaError res = cudaStreamCreate(&cudaStream);
		CHECK_CUDA_RT_ERROR(res);
		(*stream) = cudaStream;
		return NOS_RESULT_SUCCESS;
	}
	nosResult DestroyStream(nosCUDAStream stream)
	{
		cudaError res = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult CreateEvent(nosCUDAEvent* cudaEvent)
	{
		cudaEvent_t event;
		cudaError res = cudaEventCreate(&event);
		CHECK_CUDA_RT_ERROR(res);
		(*cudaEvent) = event;
		return NOS_RESULT_SUCCESS;
	}
	nosResult DestroyEvent(nosCUDAEvent cudaEvent)
	{
		cudaError res = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cudaEvent));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult LoadKernelModulePTX(const char* ptxPath, nosCUDAModule* outModule)
	{
		CUmodule cuModule;
		CUresult res = cuModuleLoad(&cuModule, ptxPath);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outModule) = cuModule;
		return NOS_RESULT_SUCCESS;
	}
	nosResult GetModuleKernelFunction(const char* functionName, nosCUDAModule* cudaModule, nosCUDAKernelFunction* outFunction)
	{
		CUfunction cuFunction;
		CUresult res = cuModuleGetFunction(&cuFunction, reinterpret_cast<CUmodule>(cudaModule), functionName);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outFunction) = cuFunction;
		return NOS_RESULT_SUCCESS;
	}
	nosResult LaunchModuleKernelFunction(nosCUDAStream* stream, nosCUDAKernelFunction* outFunction, nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData)
	{
		CUresult res = cuLaunchKernel(reinterpret_cast<CUfunction>(outFunction),
			config.GridDimensions.x, config.GridDimensions.y, config.GridDimensions.z,
			config.BlockDimensions.x, config.BlockDimensions.y, config.BlockDimensions.z, 
			config.DynamicMemorySize, reinterpret_cast<CUstream>(stream), arguments, 0);
		CHECK_CUDA_DRIVER_ERROR(res);
		return AddCallback(stream, callback, callbackData);
	}
	nosResult WaitStream(nosCUDAStream* stream)
	{
		cudaError res = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult BeginStreamTimeMeasure(nosCUDAStream* stream, nosCUDAEvent* measureEvent)
	{
		cudaError res = cudaEventRecord(reinterpret_cast<cudaEvent_t>(measureEvent), reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult EndStreamTimeMeasure(nosCUDAStream* stream, float* elapsedTime, nosCUDAEvent* measureEvent)
	{
		cudaEvent_t endEvent;
		
		cudaError res = cudaEventCreate(&endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventRecord(reinterpret_cast<cudaEvent_t>(endEvent), reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventSynchronize(endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventElapsedTime(elapsedTime, reinterpret_cast<cudaEvent_t>(measureEvent), endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		return NOS_RESULT_SUCCESS;
	}
	nosResult CopyMemory(nosCUDAStream* stream, nosCUDABufferInfo* sourceBuffer, nosCUDABufferInfo* destinationBuffer, nosCUDACopyKind copyKind)
	{
		return nosResult();
	}
	nosResult AddCallback(nosCUDAStream* stream, nosCUDACallbackFunction callback, void* callbackData)
	{
		CUresult res = cuLaunchHostFunc(reinterpret_cast<CUstream>(stream), callback, callbackData);
		CHECK_CUDA_DRIVER_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult CreateOnGPU(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
	nosResult CreateShareableOnGPU(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
	nosResult CreateManaged(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
	nosResult CreatePinned(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
	nosResult Destroy(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
}
