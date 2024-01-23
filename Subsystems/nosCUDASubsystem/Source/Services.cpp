#include "Services.h"

// SDK
#include <Nodos/SubsystemAPI.h>

#include <cuda_runtime.h>
#include <cuda.h>

namespace nos::cudass 
{
	void Bind(nosCUDASubsystem* subsys) {
		subsys->Initialize = Initialize;
		subsys->GetCudaVersion = GetCudaVersion;
		subsys->GetDeviceCount = GetDeviceCount;
		subsys->GetDeviceProperties = GetDeviceProperties;

		subsys->CreateStream = CreateStream;
		subsys->DestroyStream = DestroyStream;
		subsys->CreateEvent = CreateEvent;
		subsys->DestroyEvent = DestroyEvent;
		
		subsys->LoadKernelModulePTX = LoadKernelModulePTX;
		subsys->GetModuleKernelFunction = GetModuleKernelFunction;
		subsys->LaunchModuleKernelFunction = LaunchModuleKernelFunction;
		
		subsys->WaitStream = WaitStream;
		subsys->AddEventToStream = AddEventToStream;
		subsys->WaitEvent = WaitEvent;
		subsys->QueryEvent = QueryEvent;
		subsys->GetEventElapsedTime = GetEventElapsedTime; 
		
		subsys->CopyBuffers = CopyBuffers;
		subsys->AddCallback = AddCallback;
		
		subsys->CreateOnCUDA = CreateOnCUDA;
		subsys->CreateShareableOnCUDA = CreateShareableOnCUDA;
		subsys->CreateManaged = CreateManaged;
		subsys->CreatePinned = CreatePinned;
		subsys->Create = Create;
		subsys->Destroy = Destroy;
	}

	nosResult NOSAPI_CALL Initialize(int device)
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
	nosResult NOSAPI_CALL GetCudaVersion(CUDAVersion* versionInfo)
	{
		int cudaVersion = 0;
		cudaError res = cudaDriverGetVersion(&cudaVersion);
		CHECK_CUDA_RT_ERROR(res);

		versionInfo->Major = cudaVersion / 1000;
		versionInfo->Minor = (cudaVersion - (cudaVersion / 1000)*1000)/10;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL GetDeviceCount(int* deviceCount)
	{
		cudaError res = cudaSuccess;
		res = cudaGetDeviceCount(deviceCount);
		CHECK_CUDA_RT_ERROR(res);
	}
	nosResult NOSAPI_CALL GetDeviceProperties(int device, nosCUDADeviceProperties* deviceProperties)
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
	nosResult NOSAPI_CALL CreateStream(nosCUDAStream* stream)
	{
		cudaStream_t cudaStream;
		cudaError res = cudaStreamCreate(&cudaStream);
		CHECK_CUDA_RT_ERROR(res);
		(*stream) = cudaStream;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL DestroyStream(nosCUDAStream stream)
	{
		cudaError res = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateEvent(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags)
	{
		cudaEvent_t event;
		cudaError res = cudaEventCreate(&event, flags);
		CHECK_CUDA_RT_ERROR(res);
		(*cudaEvent) = event;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL DestroyEvent(nosCUDAEvent cudaEvent)
	{
		cudaError res = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cudaEvent));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL LoadKernelModulePTX(const char* ptxPath, nosCUDAModule* outModule)
	{
		CUmodule cuModule;
		CUresult res = cuModuleLoad(&cuModule, ptxPath);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outModule) = cuModule;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL GetModuleKernelFunction(const char* functionName, nosCUDAModule cudaModule, nosCUDAKernelFunction* outFunction)
	{
		CUfunction cuFunction = NULL;;
		CUresult res = cuModuleGetFunction(&cuFunction, reinterpret_cast<CUmodule>(cudaModule), functionName);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outFunction) = cuFunction;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL LaunchModuleKernelFunction(nosCUDAStream stream, nosCUDAKernelFunction outFunction, nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData)
	{
		CUresult res = cuLaunchKernel(reinterpret_cast<CUfunction>(outFunction),
			config.GridDimensions.x, config.GridDimensions.y, config.GridDimensions.z,
			config.BlockDimensions.x, config.BlockDimensions.y, config.BlockDimensions.z, 
			config.DynamicMemorySize, reinterpret_cast<CUstream>(stream), arguments, 0);
		CHECK_CUDA_DRIVER_ERROR(res);
		return AddCallback(stream, callback, callbackData);
	}
	nosResult NOSAPI_CALL WaitStream(nosCUDAStream stream)
	{
		cudaError res = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL AddEventToStream(nosCUDAStream stream, nosCUDAEvent measureEvent)
	{
		cudaError res = cudaEventRecord(reinterpret_cast<cudaEvent_t>(measureEvent), reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL WaitEvent(nosCUDAEvent waitEvent)
	{
		cudaError res = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(waitEvent));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL QueryEvent(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus)
	{
		cudaError res = cudaEventQuery(reinterpret_cast<cudaEvent_t>(waitEvent));

		if (res == cudaErrorNotReady) {
			(*eventStatus) = EVENT_STATUS_NOT_READY;
			return NOS_RESULT_SUCCESS;
		}
		else if (res == cudaSuccess) {
			(*eventStatus) = EVENT_STATUS_READY;
			return NOS_RESULT_SUCCESS;
		}
		
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL GetEventElapsedTime(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime)
	{
		cudaEvent_t endEvent;
		
		cudaError res = cudaEventCreate(&endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventRecord(reinterpret_cast<cudaEvent_t>(endEvent), reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventSynchronize(endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		res = cudaEventElapsedTime(elapsedTime, reinterpret_cast<cudaEvent_t>(theEvent), endEvent);
		CHECK_CUDA_RT_ERROR(res);
		
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CopyBuffers(nosCUDABufferInfo* source, nosCUDABufferInfo* destination)
	{
		CHECK_VALID_ARGUMENT(source);
		CHECK_VALID_ARGUMENT(destination);

		if (source->MemoryType == MEMORY_TYPE_UNREGISTERED || destination->MemoryType == MEMORY_TYPE_UNREGISTERED) {
			nosEngine.LogE("Invalid memory type for CUDA memcopy operation.");
			return NOS_RESULT_FAILED;
		}
		if (source->CreateInfo.Size != destination->CreateInfo.Size) {
			nosEngine.LogW("nosCUDABuffers have size mismatch, trimming will be performed for copying.");
		}

		cudaError res = cudaSuccess;
		size_t properSize = std::min(source->CreateInfo.Size, destination->CreateInfo.Size);
		switch (source->MemoryType) {
			case MEMORY_TYPE_HOST:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyHostToDevice);
				}
				else{
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyHostToHost);
				}
				break;
			case MEMORY_TYPE_DEVICE:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyDeviceToDevice);
				}
				else{
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyDeviceToHost);
				}
				break;
			case MEMORY_TYPE_MANAGED:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyHostToDevice);
				}
				else {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), properSize, cudaMemcpyHostToHost);
				}
				break;
			default:
				break;
		}
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL AddCallback(nosCUDAStream stream, nosCUDACallbackFunction callback, void* callbackData)
	{
		CUresult res = cuLaunchHostFunc(reinterpret_cast<CUstream>(stream), callback, callbackData);
		CHECK_CUDA_DRIVER_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateOnCUDA(nosCUDABufferInfo* cudaBuffer)
	{
		uint64_t addr = NULL;
		cudaError_t res = cudaMalloc((void**)&addr, cudaBuffer->CreateInfo.Size);
		CHECK_CUDA_RT_ERROR(res);
		cudaBuffer->Address = addr;
		cudaBuffer->ShareableHandle = NULL;
		cudaBuffer->CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_DEVICE;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateShareableOnCUDA(nosCUDABufferInfo* cudaBuffer)
	{
		return NOS_RESULT_SUCCESS;
		//CUcontext ctx;
		//CUdevice dev;
		//int supportsVMM = 0, supportsWin32 = 0;

		//cuDevicePrimaryCtxRetain(&ctx, 0);
		//cuCtxSetCurrent(ctx);
		//cuCtxGetDevice(&dev);

		//cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
		//assert(supportsVMM == 1);

		//cuDeviceGetAttribute(&supportsWin32,
		//	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, dev);
		//assert(supportsWin32 == 1);

		//CUresult status = CUDA_SUCCESS;
		//CUmemAllocationProp prop;

		//memset(&prop, 0, sizeof(prop));
		//SECURITY_ATTRIBUTES lps = {};
		//lps.nLength = sizeof(SECURITY_ATTRIBUTES);
		//lps.lpSecurityDescriptor = NULL;  // Use a NULL security descriptor for default security settings.
		//lps.bInheritHandle = TRUE;       // Set to TRUE if you want the handle to be inheritable.


		//prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		//prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		//prop.location.id = (int)dev;
		//prop.win32HandleMetaData = &lps;
		//prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
		//security::getDefaultSecurityDescriptor(&prop);

		//size_t chunk_sz;
		//status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		//assert(status == CUDA_SUCCESS);
		//const size_t aligned_size = ((size + chunk_sz - 1) / chunk_sz) * chunk_sz;

		//CUmemGenericAllocationHandle handle;

		//status = cuMemCreate(&handle, aligned_size, &prop, 0);
		//assert(status == CUDA_SUCCESS);

		//CUdeviceptr new_ptr = 0ULL;
		//status = cuMemAddressReserve(&new_ptr, (aligned_size), 0ULL, 0ULL, 0ULL);
		//assert(status == CUDA_SUCCESS);

		//status = cuMemMap(new_ptr, aligned_size, 0, handle, 0);
		//assert(status == CUDA_SUCCESS);

		//CUmemAccessDesc accessDesc = {};
		//accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		//accessDesc.location.id = dev;
		//accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		//// Make the address accessible
		//status = cuMemSetAccess(new_ptr, aligned_size, &accessDesc, 1);
		//assert(status == CUDA_SUCCESS);



		//uint64_t shareableHandle = 0;
		//status = cuMemExportToShareableHandle((void*)&shareableHandle, handle, CU_MEM_HANDLE_TYPE_WIN32, 0);
		//const char* errorMsg;
		//cuGetErrorString(status, &errorMsg);
		//assert(status == CUDA_SUCCESS);
	}
	nosResult NOSAPI_CALL CreateManaged(nosCUDABufferInfo* cudaBuffer)
	{
		uint64_t addr = NULL;
		cudaError res = cudaMallocManaged((void**)&addr, cudaBuffer->CreateInfo.Size);
		CHECK_CUDA_RT_ERROR(res);
		cudaBuffer->Address = addr;
		cudaBuffer->ShareableHandle = NULL;
		cudaBuffer->CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_MANAGED;

		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreatePinned(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
	nosResult InitBuffer(void* source, uint64_t size, nosCUDAMemoryType type, nosCUDABufferInfo* destination)
	{
		destination->Address = reinterpret_cast<uint64_t>(source);
		destination->CreateInfo.Size = size;
		destination->CreateHandle = NULL;
		destination->ShareableHandle = NULL;
		destination->MemoryType = type;

		return NOS_RESULT_SUCCESS;
	}
	nosResult Create(nosCUDABufferInfo* cudaBuffer)
	{
		void* data = malloc(cudaBuffer->CreateInfo.Size);

		cudaBuffer->Address = reinterpret_cast<uint64_t>(data);
		cudaBuffer->ShareableHandle = NULL;
		cudaBuffer->CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_HOST;

		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL Destroy(nosCUDABufferInfo* cudaBuffer)
	{
		return nosResult();
	}
}
