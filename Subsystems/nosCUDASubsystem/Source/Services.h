#pragma once
#include "nosCUDASubsystem.h"

#define CHECK_CUDA_RT_ERROR(cudaErr)	\
	do{							\
		if (cudaErr != cudaSuccess) {	\
			nosEngine.LogE("CUDA RT failed with error: %s", cudaGetErrorString(cudaErr));	\
			return NOS_RESULT_FAILED; \
		}						\
	}while(0)					\

#define CHECK_CUDA_DRIVER_ERROR(cuRes)	\
	do{							\
		if (cuRes != CUDA_SUCCESS) {	\
			const char* errorStr = nullptr; \
			cuGetErrorString(cuRes, &errorStr); \
			if(errorStr != nullptr){\
				nosEngine.LogE("CUDA Driver failed with error: %s", errorStr);	\
			}\
			else{\
				nosEngine.LogE("CUDA Driver failed with unknown error.");\
			}\
			return NOS_RESULT_FAILED; \
		}						\
	}while(0)					\

#define CHECK_VALID_ARGUMENT(ptrArg)	\
	do{							\
		if (ptrArg == nullptr) {	\
			return NOS_RESULT_FAILED; \
		}						\
	}while(0)					\

namespace nos::cudass
{
	void Bind(nosCUDASubsystem* subsys);

	nosResult Initialize(int device); //Initialize CUDA Runtime
	nosResult GetCudaVersion(CUDAVersion* versionInfo); //CUDA version
	nosResult GetDeviceCount(int* deviceCount); //Number of GPUs
	nosResult GetDeviceProperties(int device, nosCUDADeviceProperties* deviceProperties); //device cant exceed deviceCount
	
	nosResult CreateStream(nosCUDAStream* stream);
	nosResult DestroyStream(nosCUDAStream stream);

	nosResult CreateEvent(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags);
	nosResult DestroyEvent(nosCUDAEvent cudaEvent);
	
	nosResult LoadKernelModulePTX(const char* ptxPath, nosCUDAModule* outModule); //Loads .ptx files only. //TODO: This should be extended to char arrays and .cu files.
	nosResult GetModuleKernelFunction(const char* functionName, nosCUDAModule cudaModule, nosCUDAKernelFunction* outFunction);
	
	nosResult LaunchModuleKernelFunction(nosCUDAStream stream, nosCUDAKernelFunction outFunction, nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData);
	nosResult WaitStream(nosCUDAStream stream); //Waits all commands in the stream to be completed
	nosResult AddEventToStream(nosCUDAStream stream, nosCUDAEvent measureEvent);
	nosResult WaitEvent(nosCUDAEvent waitEvent);
	nosResult QueryEvent(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus); //Must be called before enqueueing the operation to stream
	nosResult GetEventElapsedTime(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime); //Get elapsed time between now and the measureEvent

	nosResult CopyMemory(nosCUDAStream stream, nosCUDABufferInfo* sourceBuffer, nosCUDABufferInfo* destinationBuffer, nosCUDACopyKind copyKind);
	nosResult AddCallback(nosCUDAStream stream, nosCUDACallbackFunction callback, void* callbackData);

	nosResult CreateOnGPU(nosCUDABufferInfo* cudaBuffer);
	nosResult CreateShareableOnGPU(nosCUDABufferInfo* cudaBuffer); //Exportable
	nosResult CreateManaged(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space 
	nosResult CreatePinned(nosCUDABufferInfo* cudaBuffer); //Pinned memory in RAM 
	nosResult Destroy(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space
}
