#pragma once
#include "nosCUDASubsystem.h"

#define CHECK_CUDA_RT_ERROR(cudaRes)	\
	do{							\
		if (cudaRes != cudaSuccess) {	\
			nosEngine.LogE("CUDA RT failed with error: %s", cudaGetErrorString(cudaRes));	\
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
	nosResult Initialize(int device); //Initialize CUDA Runtime
	nosResult GetCudaVersion(CUDAVersion* versionInfo); //CUDA version
	nosResult GetDeviceCount(int* deviceCount); //Number of GPUs
	nosResult GetDeviceProperties(int device, nosCUDADeviceProperties* deviceProperties); //device cant exceed deviceCount
	nosResult CreateStream(nosCUDAStream* stream);
	nosResult DestroyStream(nosCUDAStream stream);
	nosResult LoadKernelModulePTX(const char* ptxPath, nosCUDAModule* outModule); //Loads .ptx files only. //TODO: This should be extended to char arrays and .cu files.
	nosResult GetModuleKernelFunction(const char* functionName, nosCUDAModule* cudaModule, nosCUDAKernelFunction* outFunction);
	nosResult LaunchModuleKernelFunction(nosCUDAStream* stream, nosCUDAKernelFunction* outFunction,/*int ShouldRecordTime, */ nosCUDACallbackFunction callback);
	nosResult WaitStream(nosCUDAStream* stream); //Waits all commands in the stream to be completed
	nosResult BeginStreamTimeMeasure(nosCUDAStream* stream);
	nosResult EndStreamTimeMeasure(nosCUDAStream* stream, float* elapsedTime); //
	nosResult CopyMemory(nosCUDAStream* stream, nosCUDABufferInfo* sourceBuffer, nosCUDABufferInfo* destinationBuffer, nosCUDACopyKind copyKind);
	nosResult CreateOnGPU(nosCUDABufferInfo* cudaBuffer);
	nosResult CreateShareableOnGPU(nosCUDABufferInfo* cudaBuffer); //Exportable
	nosResult CreateManaged(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space 
	nosResult CreatePinned(nosCUDABufferInfo* cudaBuffer); //Pinned memory in RAM 
	nosResult Destroy(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space
}
