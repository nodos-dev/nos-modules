// Copyright MediaZ AS. All Rights Reserved.

#pragma once
#include "nosCUDASubsystem/nosCUDASubsystem.h"

#define CHECK_CUDA_RT_ERROR(cudaErr)	\
	do{							\
		if (cudaErr != cudaSuccess) {	\
			nosEngine.LogE("CUDA RT failed with error: %s from line: %d", cudaGetErrorString(cudaErr), __LINE__);	\
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

#define CHECK_IS_SUPPORTED(result, propertyName)	\
	do{ \
		if (result != 1) {\
			nosEngine.LogE("Property %s is not suported by current CUDA version.",#propertyName); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

#define CHECK_CONTEXT_SWITCH() \
	do{\
	   ContextSwitch();\
	} while (0); \

namespace nos::cudass
{
	void Bind(nosCUDASubsystem* subsys);


	nosResult CreateCUDAContext(nosCUDAContext* cudaContext, int device, nosCUDAContextFlags flags);
	nosResult DestroyCUDAContext(nosCUDAContext cudaContext); // Use with caution! Destroys the context

	nosResult Initialize(int device); //Initialize CUDA Runtime

	nosResult SetContext(nosCUDAContext cudaContext); //Sets and initializes the given CUDA Context for the calling module as current for any subsequent calls from that module
	nosResult SetCurrentContextToPrimary(); //Sets the current context as the primary context of CUDA Subsystem
	
	nosResult GetCurrentContext(nosCUDAContext* cudaContext); //Retrieve the primary CUDA Context of CUDA Subsystem
	nosResult GetCudaVersion(CUDAVersion* versionInfo); //CUDA version
	nosResult GetDeviceCount(int* deviceCount); //Number of GPUs
	nosResult GetDeviceProperties(int device, nosCUDADeviceProperties* deviceProperties); //device cant exceed deviceCount
	
	nosResult CreateStream(nosCUDAStream* stream);
	nosResult DestroyStream(nosCUDAStream stream);

	nosResult CreateCUDAEvent(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags);
	nosResult DestroyCUDAEvent(nosCUDAEvent cudaEvent);
	
	nosResult LoadKernelModuleFromPTX(const char* ptxPath, nosCUDAModule* outModule); //Loads .ptx files only. //TODO: This should be extended to char arrays and .cu files.
	nosResult GetModuleKernelFunction(const char* functionName, nosCUDAModule cudaModule, nosCUDAKernelFunction* outFunction);
	
	nosResult LaunchModuleKernelFunction(nosCUDAStream stream, nosCUDAKernelFunction outFunction, nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData);
	
	nosResult WaitStream(nosCUDAStream stream); //Waits all commands in the stream to be completed
	nosCUDAError QueryStream(nosCUDAStream stream); //

	nosCUDAError GetLastError(); //Waits all commands in the stream to be completed

	nosResult AddEventToStream(nosCUDAStream stream, nosCUDAEvent measureEvent);
	nosResult WaitCUDAEvent(nosCUDAEvent waitEvent);
	nosResult QueryCUDAEvent(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus); //Must be called before enqueueing the operation to stream
	nosResult GetCUDAEventElapsedTime(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime); //Get elapsed time between now and the measureEvent

	nosResult CopyBuffers(nosCUDABufferInfo* source, nosCUDABufferInfo* destination);
	nosResult AddCallback(nosCUDAStream stream, nosCUDACallbackFunction callback, void* callbackData);
	nosResult WaitExternalSemaphore(nosCUDAStream stream, nosCUDAExtSemaphore extSem);
	nosResult SignalExternalSemaphore(nosCUDAStream stream, nosCUDAExtSemaphore extSem);

	nosResult CreateBufferOnCUDA(nosCUDABufferInfo* cudaBuffer, uint64_t size);
	nosResult CreateShareableBufferOnCUDA(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Exportable
	nosResult CreateBufferOnManagedMemory(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Allocates in Unified Memory Space 
	nosResult CreateBufferPinned(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Pinned memory in RAM 
	nosResult InitBuffer(void* source, uint64_t size, nosCUDAMemoryType type, nosCUDABufferInfo* destination);
	nosResult CreateBuffer(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Pinned memory in RAM 
	nosResult GetCUDABufferFromAddress(uint64_t address, nosCUDABufferInfo* outBuffer); //In case you lost the cuda buffer (hope not)

	nosResult DestroyBuffer(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space

	nosResult ImportExternalMemoryAsCUDABuffer(uint64_t Handle, size_t BlockSize, size_t AllocationSize, size_t Offset, nosCUDAExternalMemoryHandleType handleType, nosCUDABufferInfo* outBuffer);
	nosResult ImportExternalSemaphore(uint64_t handle, nosCUDAExternalSemaphoreHandleType handleType, nosCUDAExtSemaphore* extSem);

	nosResult ContextSwitch();
	nosResult GetCallingModuleID(uint64_t* ID);
}
