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

typedef void* nosCUDAContext;

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

typedef enum nosCUDAContextFlags {
	CTX_FLAG_SCHED_AUTO = 0x00,
	CTX_FLAG_SCHED_SPIN = 0x01,
	CTX_FLAG_SCHED_YIELD = 0x02,
	CTX_FLAG_SCHED_BLOCKING_SYNC = 0x04,
	CTX_FLAG_SCHED_MASK = 0x07,
	CTX_FLAG_LMEM_RESIZE_TO_MAX = 0X10,
	CTX_FLAG_COREDUMP_ENABLE = 0x20,
	CTX_FLAG_USER_COREDUMP_ENABLE = 0x40,
	CTX_FLAG_SYNC_MEMOPS = 0x80,
} nosCUDAContextFlags;

typedef enum nosCUDAError {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_MEMORY_ALLOCATION = 2,
    CUDA_ERROR_INITIALIZATIO_NERROR = 3,
    CUDA_ERROR_CUDA_RT_UNLOADING = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_INVALID_CONFIGURATION = 9,
    CUDA_ERROR_INVALID_PITCH_VALUE = 12,
    CUDA_ERROR_INVALID_SYMBOL = 13,
    CUDA_ERROR_INVALID_HOST_POINTER = 16,
    CUDA_ERROR_INVALID_DEVICE_POINTER = 17,
    CUDA_ERROR_INVALID_TEXTURE = 18,
    CUDA_ERROR_INVALID_TEXTURE_BINDING = 19,
    CUDA_ERROR_INVALID_CHANNEL_DESCRIPTOR = 20,
    CUDA_ERROR_INVALID_MEMCPYDIRECTION = 21,
    CUDA_ERROR_ADDRESS_OF_CONSTANT = 22,
    CUDA_ERROR_TEXTURE_FETCH_FAILED = 23,
    CUDA_ERROR_TEXTURE_NOT_BOUND = 24,
    CUDA_ERROR_SYNCHRONIZATION_ERROR = 25,
    CUDA_ERROR_INVALID_FILTER_SETTING = 26,
    CUDA_ERROR_INVALID_NORM_SETTING = 27,
    CUDA_ERROR_MIXED_DEVICE_EXECUTION = 28,
    CUDA_ERROR_NOT_YET_IMPLEMENTED = 31,
    CUDA_ERROR_MEMORY_VALUE_TOO_LARGE = 32,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_INSUFFICIENT_DRIVER = 35,
    CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
    CUDA_ERROR_INVALID_SURFACE = 37,
    CUDA_ERROR_DUPLICATE_VARIABLENAME = 43,
    CUDA_ERROR_DUPLICATE_TEXTURENAME = 44,
    CUDA_ERROR_DUPLICATE_SURFACENAME = 45,
    CUDA_ERROR_DEVICES_UNAVAILABLE = 46,
    CUDA_ERROR_INCOMPATIBLE_DRIVERCONTEXT = 49,
    CUDA_ERROR_MISSING_CONFIGURATION = 52,
    CUDA_ERROR_PRIOR_LAUNCH_FAILURE = 53,
    CUDA_ERROR_LAUNCH_MAX_DEPTH_EXCEEDED = 65,
    CUDA_ERROR_LAUNCH_FILE_SCOPED_TEX = 66,
    CUDA_ERROR_LAUNCH_FILE_SCOPED_SURF = 67,
    CUDA_ERROR_SYNC_DEPTH_EXCEEDED = 68,
    CUDA_ERROR_LAUNCH_PENDING_COUNT_EXCEEDED = 69,
    CUDA_ERROR_INVALID_DEVICE_FUNCTION = 98,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_SOFTWARE_VALIDITY_NOT_ESTABLISHED = 103,
    CUDA_ERROR_STARTUP_FAILURE = 127,
    CUDA_ERROR_INVALID_KERNEL_IMAGE = 200,
    CUDA_ERROR_DEVICE_UNINITIALIZED = 201,
    CUDA_ERROR_MAP_BUFFER_OBJECT_FAILED = 205,
    CUDA_ERROR_UNMAP_BUFFER_OBJECT_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_KERNELIMAGE_FOR_DEVICE = 209,
    CUDA_ERROR_ALREADYA_CQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_DEVICE_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UN_SUPPORTED_EXECAFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_IN_IT_FAILED = 303,
	CUDA_ERROR_OPERATING_SYSTEM = 304,
	CUDA_ERROR_INVALID_RESOURCE_HANDLE = 400,
	CUDA_ERROR_ILLEGAL_STATE = 401,
	CUDA_ERROR_SYMBOL_NOT_FOUND = 500,
	CUDA_ERROR_NOT_READY = 600,
	CUDA_ERROR_ILLEGAL_ADDRESS = 700,
	CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
	CUDA_ERROR_LAUNCH_TIMEOUT = 702,
	CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
	CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
	CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
	CUDA_ERROR_SET_ON_ACTIVE_PROCESS = 708,
	CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
	CUDA_ERROR_ASSERT = 710,
	CUDA_ERROR_TOO_MANY_PEERS = 711,
	CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
	CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
	CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
	CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
	CUDA_ERROR_MISALIGNED_ADDRESS = 716,
	CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
	CUDA_ERROR_INVALID_PC = 718,
	CUDA_ERROR_LAUNCH_FAILURE = 719,
	CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
	CUDA_ERROR_NOT_PERMITTED = 800,
	CUDA_ERROR_NOT_SUPPORTED = 801,
	CUDA_ERROR_SYSTEM_NOT_READY = 802,
	CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
	CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
	CUDA_ERROR_MPSCONNECTION_FAILED = 805,
	CUDA_ERROR_MPS_RPC_FAILURE = 806,
	CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
	CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
	CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
	CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
	CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
	CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
	CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
	CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
	CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
	CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
	CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
	CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
	CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
	CUDA_ERROR_CAPTURED_EVENT = 907,
	CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
	CUDA_ERROR_TIMEOUT = 909,
	CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
	CUDA_ERROR_EXTERNAL_DEVICE = 911,
	CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
	CUDA_ERROR_UNKNOWN = 999,
	CUDA_ERROR_API_FAILURE_BASE = 10000,
} nosCUDAError;
#pragma endregion

#pragma region Structs

typedef struct CUDAVersion {
	unsigned short Major;
	unsigned short Minor;
} CUDAVersion;

typedef struct nosCUDABufferCreateInfo {
	uint64_t Offset;
	uint64_t AllocationSize;
	uint64_t BlockSize;
	bool IsImported;
}nosCUDABufferCreateInfo;

typedef struct nosCUDABufferShareInfo {
	uint64_t ShareableHandle;
	uint64_t CreateHandle; //Only should be used in cuMemRelease(CreateHandle) 
}nosCUDABufferShareInfo;

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
	
	nosResult(NOSAPI_CALL* CreateCUDAContext)(nosCUDAContext* cudaContext, int device, nosCUDAContextFlags flags); 
	nosResult(NOSAPI_CALL* DestroyCUDAContext)(nosCUDAContext cudaContext); // Use with caution! Destroys the context

	nosResult(NOSAPI_CALL* Initialize)(int device); //Initialize CUDA Runtime
	nosResult(NOSAPI_CALL* GetCurrentContext)(nosCUDAContext* cudaContext); //Retrieve the primary CUDA Context of CUDA Subsystem

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

	nosCUDAError(NOSAPI_CALL* GetLastError)(); //Waits all commands in the stream to be completed

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