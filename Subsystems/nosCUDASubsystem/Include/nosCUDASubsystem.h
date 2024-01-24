#ifndef NOS_CUDA_SUBSYSTEM_H_INCLUDED
#define NOS_CUDA_SUBSYSTEM_H_INCLUDED

#include "Nodos/Types.h"

#pragma region Type Definitions

typedef void* nosCUDAStream;

typedef void* nosCUDAModule;

typedef void* nosCUDAKernelFunction;

typedef void* nosCUDAEvent;

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

#pragma endregion

#pragma region Structs

typedef struct CUDAVersion {
	unsigned short Major;
	unsigned short Minor;
} CUDAVersion;

typedef struct nosCUDABufferCreateInfo {
	uint64_t Size;
}nosCUDABufferCreateInfo;

typedef struct nosCUDABufferInfo {
	nosCUDABufferCreateInfo CreateInfo;
	uint64_t Address;
	uint64_t ShareableHandle;
	uint64_t CreateHandle;
	nosCUDAMemoryType MemoryType;
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

	nosResult(NOSAPI_CALL* CreateEvent)(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags);
	nosResult(NOSAPI_CALL* DestroyEvent)(nosCUDAEvent cudaEvent);

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
	nosResult(NOSAPI_CALL* WaitEvent)(nosCUDAEvent waitEvent); //Must be called before enqueueing the operation to stream
	nosResult(NOSAPI_CALL* QueryEvent)(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus); //Must be called before enqueueing the operation to stream
	nosResult(NOSAPI_CALL* GetEventElapsedTime)(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime); //
	nosResult(NOSAPI_CALL* AddCallback)(nosCUDAStream stream, nosCUDACallbackFunction callback, void* callbackData);

	nosResult(NOSAPI_CALL* CreateOnCUDA)(nosCUDABufferInfo* cudaBuffer); //CUDA Memory, can be used in kernels etc.
	nosResult(NOSAPI_CALL* CreateShareableOnCUDA)(nosCUDABufferInfo* cudaBuffer); //Exportable CUDA memory
	nosResult(NOSAPI_CALL* CreateManaged)(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space 
	nosResult(NOSAPI_CALL* CreatePinned)(nosCUDABufferInfo* cudaBuffer); //Allocates Pinned(page-locked) memory in RAM
	nosResult(NOSAPI_CALL* Create)(nosCUDABufferInfo* cudaBuffer); //Allocates memory in RAM
	nosResult(NOSAPI_CALL* InitBuffer)(void* source, uint64_t size, nosCUDAMemoryType type ,nosCUDABufferInfo* destination); //               Wraps buffer to an externally created memory
	nosResult(NOSAPI_CALL* CopyBuffers)(nosCUDABufferInfo* source, nosCUDABufferInfo* destination);

	nosResult(NOSAPI_CALL* Destroy)(nosCUDABufferInfo* cudaBuffer); //Free the memory

	//TODO: Add semaphore stuff
	//TODO: Add texture & surface memory
	// Texture and Surface memory may not be necessary at all because Pytorch seems like using linear cuda memory:
	// https://pytorch.org/docs/stable/notes/cuda.html

} nosCUDASubsystem;

extern nosCUDASubsystem* nosCUDA;
#define NOS_CUDA_SUBSYSTEM_NAME "nos.sys.cuda"
#endif //NOS_CUDA_SUBSYSTEM_H_INCLUDED