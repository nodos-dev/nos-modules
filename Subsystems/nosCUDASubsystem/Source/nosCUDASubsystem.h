#ifndef NOS_CUDA_SUBSYSTEM_H_INCLUDED
#define NOS_CUDA_SUBSYSTEM_H_INCLUDED

#include "Nodos/Types.h"

#pragma region Type Definitions

typedef void* nosCUDAStream;

typedef void* nosCUDAModule;

typedef void* nosCUDAKernelFunction;

#ifdef _WIN32
#define NOS_CUDA_CALLBACK __stdcall
#else
#define NOS_CUDA_CALLBACK
#endif

typedef void (NOS_CUDA_CALLBACK* nosCUDACallbackFunction)(void* data);

#define NOS_CUDA_SUBSYSTEM_NAME "nos.sys.cuda"
#define UUID_SIZE 16
#define DEVICE_NAME_SIZE 256
#define WARP_SIZE 32 //Number of threads should be choosen as a multiple of Warp Size, whenever possible

#pragma endregion

#pragma region Structs

typedef struct CUDAVersion {
	unsigned short Major;
	unsigned short Minor;
} CUDAVersion;

typedef struct nosCUDABufferInfo {
	uint64_t Address = NULL;
	uint64_t ShareableHandle = NULL;
	uint64_t Size = 0;
	uint64_t CreateHandle = NULL;
	nosCUDAMemoryType memoryType = MEMORY_TYPE_UNREGISTERED;
}nosCUDAResourceInfo;

typedef struct nosCUDAArrayInfo {

} nosCUDAArrayInfo;

typedef struct nosCUDADeviceProperties {
	char Name[256];
	char DeviceUUID[16];
	uint32_t ComputeCapabilityMajor;
	uint32_t ComputeCapabilityMinor;
} nosCudaDeviceProperties;

typedef struct nosCUDALaunchConfig {
	nosDim3 GridDimensions;
	nosDim3 BlockDimensions;  

} nosCUDALaunchConfig;

typedef struct nosDim3 {
	int x;
	int y;
	int z;
} nosDim3;
#pragma endregion

#pragma region Enums
typedef enum nosCUDACopyKind {
	HOST_TO_HOST		= 0,			
	HOST_TO_DEVICE		= 1,			
	DEVICE_TO_HOST		= 2,			
	DEVICE_TO_DEVICE	= 3,			
	//TODO: Add MEMCPY_DEFAULT like CUDA			 
} nosCUDACopyKind;

typedef enum nosCUDAMemoryType
{
	MEMORY_TYPE_UNREGISTERED	= 0, /**< Unregistered memory */
	MEMORY_TYPE_HOST			= 1, /**< Host memory */
	MEMORY_TYPE_DEVICE			= 2, /**< Device memory */
	MEMORY_TYPE_MANAGED			= 3  /**< Managed memory */
} nosCUDAMemoryType;

#pragma endregion

typedef struct nosCUDASubsystem
{

	nosResult(NOSAPI_CALL* Initialize)(int device); //Initialize CUDA Runtime
	nosResult(NOSAPI_CALL* GetCudaVersion)(CUDAVersion* versionInfo); //CUDA version
	nosResult(NOSAPI_CALL* GetDeviceCount)(int* deviceCount); //Number of GPUs
	nosResult(NOSAPI_CALL* GetDeviceProperties)(int device, nosCUDADeviceProperties* deviceProperties); //device cant exceed deviceCount

	nosResult(NOSAPI_CALL* CreateStream)(nosCUDAStream* stream);
	nosResult(NOSAPI_CALL* DestroyStream)(nosCUDAStream stream);

	nosResult(NOSAPI_CALL* LoadKernelModulePTX)(const char* ptxPath, nosCUDAModule* outModule); //Loads .ptx files only. //TODO: This should be extended to char arrays and .cu files.
	nosResult(NOSAPI_CALL* GetModuleKernelFunction)(const char* functionName, nosCUDAModule* cudaModule, nosCUDAKernelFunction* outFunction);

	nosResult(NOSAPI_CALL* LaunchModuleKernelFunction)(nosCUDAStream* stream, nosCUDAKernelFunction* outFunction,/*int ShouldRecordTime, */ nosCUDACallbackFunction callback);
	nosResult(NOSAPI_CALL* WaitStream)(nosCUDAStream* stream); //Waits all commands in the stream to be completed
	nosResult(NOSAPI_CALL* BeginStreamTimeMeasure)(nosCUDAStream* stream);
	nosResult(NOSAPI_CALL* EndStreamTimeMeasure)(nosCUDAStream* stream, float* elapsedTime); //
	nosResult(NOSAPI_CALL* CopyMemory)(nosCUDAStream* stream, nosCUDABufferInfo* sourceBuffer, nosCUDABufferInfo* destinationBuffer, nosCUDACopyKind copyKind);

	nosResult(NOSAPI_CALL* CreateOnGPU)(nosCUDABufferInfo* cudaBuffer);
	nosResult(NOSAPI_CALL* CreateShareableOnGPU)(nosCUDABufferInfo* cudaBuffer); //Exportable
	nosResult(NOSAPI_CALL* CreateManaged)(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space 
	nosResult(NOSAPI_CALL* CreatePinned)(nosCUDABufferInfo* cudaBuffer); //Pinned memory in RAM 
	nosResult(NOSAPI_CALL* Destroy)(nosCUDABufferInfo* cudaBuffer); //Allocates in Unified Memory Space

	//TODO: Add semaphore stuff
	//TODO: Add texture & surface memory

} nosCUDASubsystem;


#endif //NOS_CUDA_SUBSYSTEM_H_INCLUDED