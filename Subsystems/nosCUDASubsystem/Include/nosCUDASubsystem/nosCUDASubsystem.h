#pragma once
#ifndef NOS_CUDA_SUBSYSTEM_H_INCLUDED
#define NOS_CUDA_SUBSYSTEM_H_INCLUDED

#include "Nodos/Types.h"

#pragma region Type Definitions

typedef void* nosCUDAStream;

typedef void* nosCUDAModule;

typedef void* nosCUDAKernelFunction;

typedef void* nosCUDAEvent;

typedef void* nosCUDAExtSemaphore;

#ifdef _WIN32
#define NOS_CUDA_CALLBACK __stdcall
#else
#define NOS_CUDA_CALLBACK
#endif

//Host(CPU) function must NOT make any CUDA API calls
typedef void (NOS_CUDA_CALLBACK* nosCUDACallbackFunction)(void* data);

#define NOS_CUDA_SUBSYSTEM_NAME "nos.sys.cuda"
#define UUID_SIZE 16
#define DEVICE_NAME_SIZE 256
#define WARP_SIZE 32 //Number of threads should be choosen as a multiple of Warp Size, whenever possible
#define MAX_THREAD_PER_BLOCK 1024

#pragma endregion


#pragma region Enums
typedef enum nosCUDACopyKind
{
	HOST_TO_HOST = 0,
	HOST_TO_DEVICE = 1,
	DEVICE_TO_HOST = 2,
	DEVICE_TO_DEVICE = 3,
	//TODO: Add MEMCPY_DEFAULT like CUDA			 
} nosCUDACopyKind;

typedef enum nosCUDAMemoryType
{
	MEMORY_TYPE_UNREGISTERED = 0, /**< Unregistered memory */
	MEMORY_TYPE_HOST = 1, /**< Host memory */
	MEMORY_TYPE_DEVICE = 2, /**< Device memory */
	MEMORY_TYPE_MANAGED = 3  /**< Managed memory */
} nosCUDAMemoryType;

typedef enum nosCUDAEventFlags {
	EVENT_FLAG_DEFAULT = 0x00,
	EVENT_FLAG_BLOCKING_SYNC = 0x01,
	EVENT_FLAG_DISABLE_TIMING = 0x02,
} nosCUDAEventFlags;

typedef enum nosCUDAEventStatus {
	EVENT_STATUS_READY = 0,
	EVENT_STATUS_NOT_READY = 1,
} nosCUDAEventStatus;

typedef enum nosCUDAExternalMemoryHandleType{
	EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEFD = 1,
	EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32 = 2,
	EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32KMT = 3,
	EXTERNAL_MEMORY_HANDLE_TYPE_D3D12HEAP = 4,
	EXTERNAL_MEMORY_HANDLE_TYPE_D3D12RESOURCE = 5,
	EXTERNAL_MEMORY_HANDLE_TYPE_D3D11RESOURCE = 6,
	EXTERNAL_MEMORY_HANDLE_TYPE_D3D11RESOURCEKMT = 7,
	EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} nosCUDAExternalMemoryHandleType;

typedef enum nosCUDAExternalSemaphoreHandleType {
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEFD = 1,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEWIN32 = 2,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEWIN32KMT = 3,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12FENCE = 4,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11FENCE = 5,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_KEYEDMUTEX = 7,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_KEYEDMUTEXKMT = 8,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINESEMAPHOREFD = 9,
	EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINESEMAPHOREWIN32 = 10
} nosCUDAExternalSemaphoreHandleType;
#pragma endregion

#pragma region Structs

typedef struct CUDAVersion {
	unsigned short Major;
	unsigned short Minor;
} CUDAVersion;

typedef struct nosCUDABufferCreateInfo {
	uint64_t RequestedSize;
	uint64_t AllocatedSize;
	bool IsImported;
}nosCUDABufferCreateInfo;

typedef struct nosCUDABufferShareInfo {
	uint64_t ShareableHandle;
	uint64_t CreateHandle; //Only should be used in cuMemRelease(CreateHandle) 
};
typedef struct nosCUDABufferInfo {
	uint64_t Address;
	nosCUDAMemoryType MemoryType;
	nosCUDABufferShareInfo ShareInfo;
	nosCUDABufferCreateInfo CreateInfo;
}nosCUDABufferInfo;

typedef struct nosCUDAArrayInfo {

} nosCUDAArrayInfo;

typedef struct nosCUDADeviceProperties {
	char Name[256];
	char DeviceUUID[16];
	uint32_t ComputeCapabilityMajor;
	uint32_t ComputeCapabilityMinor;
} nosCudaDeviceProperties;

typedef struct nosDim3 {
	unsigned int x;
	unsigned int y;
	unsigned int z;
} nosDim3;

typedef struct nosCUDAKernelLaunchConfig {
	nosDim3 GridDimensions;
	nosDim3 BlockDimensions;
	size_t DynamicMemorySize; //Default by zero, will be used for `extern __shared__ float shared[]` type dynamically allocated arrays.
} nosCUDAKernelLaunchConfig;


typedef struct nosCUDACallbackContext {
	void* Context;
	void* Data; 
} nosCUDACallbackContext;
#pragma endregion



typedef struct nosCUDASubsystem
{
	
	nosResult(NOSAPI_CALL* Initialize)(int device); //Initialize CUDA Runtime
	nosResult(NOSAPI_CALL* GetCudaVersion)(CUDAVersion* versionInfo); //CUDA version
	nosResult(NOSAPI_CALL* GetDeviceCount)(int* deviceCount); //Number of GPUs
	nosResult(NOSAPI_CALL* GetDeviceProperties)(int device, nosCUDADeviceProperties* deviceProperties); //device cant exceed deviceCount

	nosResult(NOSAPI_CALL* CreateStream)(nosCUDAStream* stream);
	nosResult(NOSAPI_CALL* DestroyStream)(nosCUDAStream stream);

	nosResult(NOSAPI_CALL* CreateCUDAEvent)(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags);
	nosResult(NOSAPI_CALL* DestroyCUDAEvent)(nosCUDAEvent cudaEvent);

	nosResult(NOSAPI_CALL* LoadKernelModuleFromPTX)(const char* ptxPath, nosCUDAModule* outModule); //Loads .ptx files only
	//nosResult(NOSAPI_CALL* LoadKernelModuleFromSource)(const char* cuFilePath, nosCUDAModule* outModule); //Loads .ptx files only
	//nosResult(NOSAPI_CALL* LoadKernelModuleFromCString)(const char* cuFilePath, nosCUDAModule* outModule); //Loads .ptx files only
	nosResult(NOSAPI_CALL* GetModuleKernelFunction)(const char* functionName, nosCUDAModule cudaModule, nosCUDAKernelFunction* outFunction);
	
	/**
	 * @brief Runs CUDA Kernels.
	 *
	 * @param config Specificies Block and Grid dimensions, hence the number of threads.
	 * @param arguments Kernel function arguments. Can be set as void* args[] = {&param1, &param2, .., &paramN} .
	 *
	 * @return
	 */
	nosResult(NOSAPI_CALL* LaunchModuleKernelFunction)(nosCUDAStream stream, nosCUDAKernelFunction outFunction,nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData);
	
	nosResult(NOSAPI_CALL* WaitStream)(nosCUDAStream stream); //Waits all commands in the stream to be completed
	nosResult(NOSAPI_CALL* AddEventToStream)(nosCUDAStream stream, nosCUDAEvent event); //Must be called before enqueueing the operation to stream
	nosResult(NOSAPI_CALL* WaitCUDAEvent)(nosCUDAEvent waitEvent); //Must be called before enqueueing the operation to stream 
	nosResult(NOSAPI_CALL* QueryCUDAEvent)(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus); //Must be called before enqueueing the operation to stream
	nosResult(NOSAPI_CALL* GetCUDAEventElapsedTime)(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime); //
	nosResult(NOSAPI_CALL* AddCallback)(nosCUDAStream stream, nosCUDACallbackFunction callback, void* callbackData);
	nosResult(NOSAPI_CALL* WaitExternalSemaphore)(nosCUDAStream stream, nosCUDAExtSemaphore extSem);
	nosResult(NOSAPI_CALL* SignalExternalSemaphore)(nosCUDAStream stream, nosCUDAExtSemaphore extSem);

	nosResult(NOSAPI_CALL* CreateBufferOnCUDA)(nosCUDABufferInfo* cudaBuffer, uint64_t size); //CUDA Memory, can be used in kernels etc.
	nosResult(NOSAPI_CALL* CreateShareableBufferOnCUDA)(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Exportable CUDA memory
	nosResult(NOSAPI_CALL* CreateBufferOnManagedMemory)(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Allocates in Unified Memory Space 
	nosResult(NOSAPI_CALL* CreateBufferPinned)(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Allocates Pinned(page-locked) memory in RAM
	nosResult(NOSAPI_CALL* CreateBuffer)(nosCUDABufferInfo* cudaBuffer, uint64_t size); //Allocates memory in RAM
	nosResult(NOSAPI_CALL* InitBuffer)(void* source, uint64_t size, nosCUDAMemoryType type ,nosCUDABufferInfo* destination); //               Wraps buffer to an externally created memory
	nosResult(NOSAPI_CALL* CopyBuffers)(nosCUDABufferInfo* source, nosCUDABufferInfo* destination);
	nosResult(NOSAPI_CALL* GetCUDABufferFromAddress)(uint64_t address, nosCUDABufferInfo* outBuffer); //In case you lost the cuda buffer (hope not)

	nosResult(NOSAPI_CALL* DestroyBuffer)(nosCUDABufferInfo* cudaBuffer); //Free the memory
	
	//Interop
	nosResult(NOSAPI_CALL* ImportExternalMemoryAsCUDABuffer)(uint64_t Handle, size_t BlockSize, size_t AllocationSize, size_t Offset, nosCUDAExternalMemoryHandleType handleType, nosCUDABufferInfo* outBuffer);
	nosResult(NOSAPI_CALL* ImportExternalSemaphore)(uint64_t handle, nosCUDAExternalSemaphoreHandleType handleType, nosCUDAExtSemaphore* extSem);

	//TODO: Add semaphore stuff
	//TODO: Add texture & surface memory
	// Texture and Surface memory may not be necessary at all because Pytorch seems like using linear cuda memory:
	// https://pytorch.org/docs/stable/notes/cuda.html

} nosCUDASubsystem;

extern nosCUDASubsystem* nosCUDA;
#define NOS_CUDA_SUBSYSTEM_NAME "nos.sys.cuda"
#endif //NOS_CUDA_SUBSYSTEM_H_INCLUDED