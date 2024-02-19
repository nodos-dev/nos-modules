// Copyright MediaZ AS. All Rights Reserved.

#include "Services.h"
#include <cstring>
#include "CUDASubsysCommon.h"
// SDK
#include <Nodos/SubsystemAPI.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "Globals.h"

namespace nos::cudass 
{
	void Bind(nosCUDASubsystem* subsys) {

		subsys->CreateCUDAContext = CreateCUDAContext;
		subsys->DestroyCUDAContext = DestroyCUDAContext;

		subsys->Initialize = Initialize;
		subsys->GetCurrentContext = GetCurrentContext;
		subsys->GetCudaVersion = GetCudaVersion;
		subsys->GetDeviceCount = GetDeviceCount;
		subsys->GetDeviceProperties = GetDeviceProperties;

		subsys->CreateStream = CreateStream;
		subsys->DestroyStream = DestroyStream;
		subsys->CreateCUDAEvent = CreateCUDAEvent;
		subsys->DestroyCUDAEvent = DestroyCUDAEvent;

		subsys->LoadKernelModuleFromPTX = LoadKernelModuleFromPTX;
		subsys->LoadKernelModuleFromCString = LoadKernelModuleFromCString;
		subsys->GetModuleKernelFunction = GetModuleKernelFunction;
		subsys->LaunchModuleKernelFunction = LaunchModuleKernelFunction;

		subsys->WaitStream = WaitStream;
		subsys->QueryStream = QueryStream;
		
		subsys->GetLastError = GetLastError;

		subsys->AddEventToStream = AddEventToStream;
		subsys->WaitCUDAEvent = WaitCUDAEvent;
		subsys->QueryCUDAEvent = QueryCUDAEvent;
		subsys->GetCUDAEventElapsedTime = GetCUDAEventElapsedTime;
		subsys->WaitExternalSemaphore = WaitExternalSemaphore;
		subsys->SignalExternalSemaphore = SignalExternalSemaphore;

		subsys->AddCallback = AddCallback;

		subsys->CreateBufferOnCUDA = CreateBufferOnCUDA;
		subsys->CreateShareableBufferOnCUDA = CreateShareableBufferOnCUDA;
		subsys->CreateBufferOnManagedMemory = CreateBufferOnManagedMemory;
		subsys->CreateBufferPinned = CreateBufferPinned;
		subsys->CreateBuffer = CreateBuffer;
		subsys->CopyBuffers = CopyBuffers;
		subsys->GetCUDABufferFromAddress = GetCUDABufferFromAddress;

		subsys->DestroyBuffer = DestroyBuffer;
		
		subsys->ImportExternalSemaphore = ImportExternalSemaphore;
		subsys->ImportExternalMemoryAsCUDABuffer = ImportExternalMemoryAsCUDABuffer;
	}
	nosResult NOSAPI_CALL CreateCUDAContext(nosCUDAContext* cudaContext, int device, nosCUDAContextFlags flags)
	{
		uint64_t ID = NULL;
		nosResult nosRes = GetCallingModuleID(&ID);
		if (nosRes != NOS_RESULT_SUCCESS)
			return nosRes;
		CUcontext theContext = {};
		CUresult cuRes = cuCtxCreate(&theContext, flags, device);
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		cuRes = cuCtxPopCurrent(reinterpret_cast<CUcontext*>(&ActiveContext));
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		(*cudaContext) = reinterpret_cast<nosCUDAContext>(theContext);
		IDContextMap[ID] = (*cudaContext);
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL DestroyCUDAContext(nosCUDAContext cudaContext)
	{
		if (cudaContext == PrimaryContext)
			return NOS_RESULT_SUCCESS;

		CUresult cuRes = cuCtxDestroy(reinterpret_cast<CUcontext>(cudaContext));
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL Initialize(int device)
	{
		if (device == CurrentDevice && PrimaryContext != nullptr) {
			//Already initialized
			return NOS_RESULT_SUCCESS;
		}
		
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
		
		CUresult cuRes = cuCtxSetFlags(CU_CTX_COREDUMP_ENABLE);
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		cuRes = cuCtxGetCurrent(reinterpret_cast<CUcontext*>(&PrimaryContext));
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		ActiveContext = PrimaryContext;

		std::string CoreDumpFile = std::string(nosEngine.Context->RootFolderPath) + "/CoreDump.txt";
		size_t Size = CoreDumpFile.size();

		cuRes = cuCoredumpSetAttribute(CU_COREDUMP_FILE, &CoreDumpFile, &Size);
		CurrentDevice = device;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL SetContext(nosCUDAContext cudaContext)
	{
		CUresult cuRes = cuCtxSetCurrent(reinterpret_cast<CUcontext>(cudaContext));
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		ActiveContext = cudaContext;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL SetCurrentContextToPrimary()
	{
		CUresult cuRes = cuCtxSetCurrent(reinterpret_cast<CUcontext>(PrimaryContext));
		CHECK_CUDA_DRIVER_ERROR(cuRes);
		ActiveContext = PrimaryContext;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL GetCurrentContext(nosCUDAContext* cudaContext)
	{
		(*cudaContext) = reinterpret_cast<nosCUDAContext>(PrimaryContext);  
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
		CHECK_CONTEXT_SWITCH();
		cudaStream_t cudaStream;
		cudaError res = cudaStreamCreate(&cudaStream);
		CHECK_CUDA_RT_ERROR(res);
		(*stream) = cudaStream;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL DestroyStream(nosCUDAStream stream)
	{
		CHECK_CONTEXT_SWITCH();
		cudaError res = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateCUDAEvent(nosCUDAEvent* cudaEvent, nosCUDAEventFlags flags)
	{
		CHECK_CONTEXT_SWITCH();
		cudaEvent_t event;
		cudaError res = cudaEventCreate(&event, flags);
		CHECK_CUDA_RT_ERROR(res);
		(*cudaEvent) = event;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL DestroyCUDAEvent(nosCUDAEvent cudaEvent)
	{
		CHECK_CONTEXT_SWITCH();
		cudaError res = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(cudaEvent));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL LoadKernelModuleFromPTX(const char* ptxPath, nosCUDAModule* outModule)
	{
		CHECK_CONTEXT_SWITCH();
		CUmodule cuModule;
		CUresult res = cuModuleLoad(&cuModule, ptxPath);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outModule) = cuModule;
		return NOS_RESULT_SUCCESS;
	}
	nosResult LoadKernelModuleFromCString(const char* ptxString, const char* errorLogBuffer, uint64_t errorLogBufferSize, nosCUDAModule* outModule)
	{
		CHECK_CONTEXT_SWITCH();
		CUmodule cuModule;
		CUjit_option options[3];
		void* values[3];

		options[0] = CU_JIT_ERROR_LOG_BUFFER;
		values[0] = (void*)errorLogBuffer;
		options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		values[1] = (void*)errorLogBufferSize;
		options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
		values[2] = 0;

		CUresult res = cuModuleLoadDataEx(&cuModule, ptxString, 3, options, values);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outModule) = cuModule;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL GetModuleKernelFunction(const char* functionName, nosCUDAModule cudaModule, nosCUDAKernelFunction* outFunction)
	{
		CHECK_CONTEXT_SWITCH();
		CUfunction cuFunction = NULL;;
		CUresult res = cuModuleGetFunction(&cuFunction, reinterpret_cast<CUmodule>(cudaModule), functionName);
		CHECK_CUDA_DRIVER_ERROR(res);
		(*outFunction) = cuFunction;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL LaunchModuleKernelFunction(nosCUDAStream stream, nosCUDAKernelFunction outFunction, nosCUDAKernelLaunchConfig config, void** arguments, nosCUDACallbackFunction callback, void* callbackData)
	{
		CHECK_CONTEXT_SWITCH();
		CUresult res = cuLaunchKernel(reinterpret_cast<CUfunction>(outFunction),
			config.GridDimensions.x, config.GridDimensions.y, config.GridDimensions.z,
			config.BlockDimensions.x, config.BlockDimensions.y, config.BlockDimensions.z, 
			config.DynamicMemorySize, reinterpret_cast<CUstream>(stream), arguments, 0);
		CHECK_CUDA_DRIVER_ERROR(res);
		return AddCallback(stream, callback, callbackData);
	}
	nosResult NOSAPI_CALL WaitStream(nosCUDAStream stream)
	{
		CHECK_CONTEXT_SWITCH();
		cudaError res = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosCUDAError QueryStream(nosCUDAStream stream)
	{
		CHECK_CONTEXT_SWITCH();
		return static_cast<nosCUDAError>(cudaStreamQuery(reinterpret_cast<CUstream>(stream)));
	}
	nosCUDAError GetLastError()
	{
		CHECK_CONTEXT_SWITCH();
		return static_cast<nosCUDAError>(cudaGetLastError());
	}
	nosResult NOSAPI_CALL AddEventToStream(nosCUDAStream stream, nosCUDAEvent measureEvent)
	{
		CHECK_CONTEXT_SWITCH();
		cudaError res = cudaEventRecord(reinterpret_cast<cudaEvent_t>(measureEvent), reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL WaitCUDAEvent(nosCUDAEvent waitEvent)
	{
		CHECK_CONTEXT_SWITCH();
		cudaError res = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(waitEvent));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL QueryCUDAEvent(nosCUDAEvent waitEvent, nosCUDAEventStatus* eventStatus)
	{
		CHECK_CONTEXT_SWITCH();
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
	nosResult NOSAPI_CALL GetCUDAEventElapsedTime(nosCUDAStream stream, nosCUDAEvent theEvent, float* elapsedTime)
	{
		CHECK_CONTEXT_SWITCH();
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
		CHECK_CONTEXT_SWITCH();
		CHECK_VALID_ARGUMENT(source);
		CHECK_VALID_ARGUMENT(destination);

		if (source->MemoryType == MEMORY_TYPE_UNREGISTERED || destination->MemoryType == MEMORY_TYPE_UNREGISTERED) {
			nosEngine.LogE("Invalid memory type for CUDA memcopy operation.");
			return NOS_RESULT_FAILED;
		}
		if (source->CreateInfo.AllocationSize != destination->CreateInfo.AllocationSize) {
			nosEngine.LogW("nosCUDABuffers have size mismatch, trimming will be performed for copying.");
		}

		cudaError res = cudaSuccess;
		size_t safeCopySize = std::min(source->CreateInfo.AllocationSize, destination->CreateInfo.BlockSize);
		switch (source->MemoryType) {
			case MEMORY_TYPE_HOST:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyHostToDevice);
				}
				else{
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyHostToHost);
				}
				break;
			case MEMORY_TYPE_DEVICE:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyDeviceToDevice);
				}
				else{
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyDeviceToHost);
				}
				break;
			case MEMORY_TYPE_MANAGED:
				if (destination->MemoryType == MEMORY_TYPE_DEVICE) {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyHostToDevice);
				}
				else {
					res = cudaMemcpy(reinterpret_cast<void*>(destination->Address), reinterpret_cast<void*>(source->Address), safeCopySize, cudaMemcpyHostToHost);
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
		CHECK_CONTEXT_SWITCH();
		if (callback == nullptr || callbackData == nullptr)
			return NOS_RESULT_SUCCESS;

		CUresult res = cuLaunchHostFunc(reinterpret_cast<CUstream>(stream), callback, callbackData);
		CHECK_CUDA_DRIVER_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult WaitExternalSemaphore(nosCUDAStream stream, nosCUDAExtSemaphore extSem)
	{
		CHECK_CONTEXT_SWITCH();
		cudaExternalSemaphoreWaitParams params = {};
		memset(&params, 0, sizeof(params));
		cudaError res = cudaWaitExternalSemaphoresAsync(reinterpret_cast<cudaExternalSemaphore_t*>(&extSem), &params, 1, reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult SignalExternalSemaphore(nosCUDAStream stream, nosCUDAExtSemaphore extSem)
	{
		CHECK_CONTEXT_SWITCH();
		cudaExternalSemaphoreSignalParams params = {};
		memset(&params, 0, sizeof(params));
		cudaError res = cudaSignalExternalSemaphoresAsync(reinterpret_cast<cudaExternalSemaphore_t*>(&extSem), &params, 1, reinterpret_cast<cudaStream_t>(stream));
		CHECK_CUDA_RT_ERROR(res);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateBufferOnCUDA(nosCUDABufferInfo* cudaBuffer, uint64_t size)
	{
		CHECK_CONTEXT_SWITCH();
		uint64_t addr = NULL;
		cudaError_t res = cudaMalloc((void**)&addr, size);
		CHECK_CUDA_RT_ERROR(res);
		cudaBuffer->Address = addr;
		cudaBuffer->ShareInfo.ShareableHandle = NULL;
		cudaBuffer->ShareInfo.CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_DEVICE;
		cudaBuffer->CreateInfo.BlockSize = size;
		cudaBuffer->CreateInfo.AllocationSize = size;
		cudaBuffer->CreateInfo.IsImported = false;
		ResManager.Add(addr, *cudaBuffer);
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateShareableBufferOnCUDA(nosCUDABufferInfo* cudaBuffer, uint64_t size)
	{
		CHECK_CONTEXT_SWITCH();
		CUdevice dev;
		int supportsVMM = 0, supportsWin32 = 0;
		
		//cuCtxSetCurrent(reinterpret_cast<CUcontext>(PrimaryContext));
		cuCtxGetDevice(&dev);

		cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
		CHECK_IS_SUPPORTED(supportsVMM, VIRTUAL_MEMORY_MANAGEMENT);

		cuDeviceGetAttribute(&supportsWin32,
			CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, dev);
		CHECK_IS_SUPPORTED(supportsVMM, HANDLE_TYPE_WIN32_HANDLE);

		CUresult status = CUDA_SUCCESS;
		CUmemAllocationProp prop;
		
		memset(&prop, 0, sizeof(prop));

		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = (int)dev;
		prop.win32HandleMetaData = Descriptor::SecurityDescriptor::GetDefaultSecurityDescriptor();
		prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;

		size_t chunk_sz;
		status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		CHECK_CUDA_DRIVER_ERROR(status);
		const size_t aligned_size = ((size + chunk_sz - 1) / chunk_sz) * chunk_sz;

		CUmemGenericAllocationHandle allocHandle = NULL;
		
		status = cuMemCreate(&allocHandle, aligned_size, &prop, 0);
		CHECK_CUDA_DRIVER_ERROR(status);

		CUdeviceptr address = 0ULL;
		status = cuMemAddressReserve(&address, (aligned_size), 0ULL, 0ULL, 0ULL);
		CHECK_CUDA_DRIVER_ERROR(status);

		status = cuMemMap(address, aligned_size, 0, allocHandle, 0); //We should unmap this somehow after each usage and map again before usages
		CHECK_CUDA_DRIVER_ERROR(status);

		CUmemAccessDesc accessDesc = {};
		memset(&accessDesc, 0, sizeof(&accessDesc));
		accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		accessDesc.location.id = dev;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		// Make the address accessible: should we give api users to flesibiltiy to choose?
		status = cuMemSetAccess(address, aligned_size, &accessDesc, 1);
		CHECK_CUDA_DRIVER_ERROR(status);
		
		uint64_t shareableHandle = 0;
		status = cuMemExportToShareableHandle((void*)&shareableHandle, allocHandle, CU_MEM_HANDLE_TYPE_WIN32, 0);
		CHECK_CUDA_DRIVER_ERROR(status);

		cudaBuffer->CreateInfo.BlockSize = aligned_size;
		cudaBuffer->CreateInfo.AllocationSize = size;
		cudaBuffer->ShareInfo.CreateHandle = allocHandle;
		cudaBuffer->ShareInfo.ShareableHandle = shareableHandle;
		cudaBuffer->Address = address;
		cudaBuffer->MemoryType = MEMORY_TYPE_DEVICE;
		ResManager.Add(address, *cudaBuffer);
		return NOS_RESULT_SUCCESS;

	}
	nosResult NOSAPI_CALL CreateBufferOnManagedMemory(nosCUDABufferInfo* cudaBuffer, uint64_t size)
	{
		CHECK_CONTEXT_SWITCH();
		uint64_t addr = NULL;
		cudaError res = cudaMallocManaged((void**)&addr, size);
		CHECK_CUDA_RT_ERROR(res);
		cudaBuffer->Address = addr;
		cudaBuffer->ShareInfo.ShareableHandle = NULL;
		cudaBuffer->ShareInfo.CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_MANAGED;
		ResManager.Add(addr, *cudaBuffer);

		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL CreateBufferPinned(nosCUDABufferInfo* cudaBuffer, uint64_t size)
	{
		CHECK_CONTEXT_SWITCH();
		uint64_t addr = NULL;
		cudaError res = cudaMallocHost((void**)&addr, size);
		CHECK_CUDA_RT_ERROR(res);
		cudaBuffer->Address = addr;
		cudaBuffer->ShareInfo.ShareableHandle = NULL;
		cudaBuffer->ShareInfo.CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_HOST;
		ResManager.Add(addr, *cudaBuffer);

		return nosResult();
	}
	nosResult InitBuffer(void* source, uint64_t size, nosCUDAMemoryType type, nosCUDABufferInfo* destination)
	{
		destination->Address = reinterpret_cast<uint64_t>(source);
		destination->CreateInfo.AllocationSize = size;
		destination->ShareInfo.CreateHandle = NULL;
		destination->ShareInfo.ShareableHandle = NULL;
		destination->MemoryType = type;
		ResManager.Add(destination->Address, *destination);

		return NOS_RESULT_SUCCESS;
	}
	nosResult CreateBuffer(nosCUDABufferInfo* cudaBuffer, uint64_t size)
	{
		void* data = malloc(size);

		cudaBuffer->Address = reinterpret_cast<uint64_t>(data);
		cudaBuffer->ShareInfo.ShareableHandle = NULL;
		cudaBuffer->ShareInfo.CreateHandle = NULL;
		cudaBuffer->MemoryType = MEMORY_TYPE_HOST;
		cudaBuffer->CreateInfo.AllocationSize = size;
		cudaBuffer->CreateInfo.BlockSize = size;
		cudaBuffer->CreateInfo.IsImported = false;
		ResManager.Add(cudaBuffer->Address, *cudaBuffer);

		return NOS_RESULT_SUCCESS;
	}
	nosResult GetCUDABufferFromAddress(uint64_t address, nosCUDABufferInfo* outBuffer)
	{
		void* res = ResManager.Get(address);
		if (res == nullptr)
			return NOS_RESULT_FAILED;
		(*outBuffer) = *reinterpret_cast<nosCUDABufferInfo*>(res);

		return NOS_RESULT_SUCCESS;
	}
	nosResult ImportExternalMemoryAsCUDABuffer(uint64_t Handle, size_t BlockSize, size_t AllocationSize, size_t Offset, nosCUDAExternalMemoryHandleType handleType, nosCUDABufferInfo* outBuffer)
	{
		CHECK_CONTEXT_SWITCH();
		//cuCtxSetCurrent(reinterpret_cast<CUcontext>(PrimaryContext));
		cudaExternalMemory_t externalMemory;
		cudaExternalMemoryHandleDesc desc;
		memset(&desc, 0, sizeof(desc));
		switch (handleType) {
			case EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEFD:
				desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32:
				desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32KMT:
				desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_D3D12HEAP:
				desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_D3D12RESOURCE:
				desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_D3D11RESOURCE:
				desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_D3D11RESOURCEKMT:
				desc.type = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
				break;
			case EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF:
				desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
				break;
		}
		//TODO: do it in switch case
		desc.handle.win32.handle = reinterpret_cast<void*>(Handle);
		desc.handle.win32.name = NULL;
		desc.size = BlockSize;

		cudaError res = cudaImportExternalMemory(&externalMemory, &desc);
		CHECK_CUDA_RT_ERROR(res);

		void* pointer = nullptr;

		cudaExternalMemoryBufferDesc bufferDesc;
		bufferDesc.flags = 0; // must be zero
		bufferDesc.offset = Offset;
		bufferDesc.size = AllocationSize;

		res = cudaExternalMemoryGetMappedBuffer(&pointer, externalMemory, &bufferDesc);
		CHECK_CUDA_RT_ERROR(res);

		//Clean resources in case of re-importing
		//DestroyBuffer(outBuffer);

		uint64_t outCudaPointerAddres = NULL;
		outBuffer->Address = reinterpret_cast<uint64_t>(pointer);
		outBuffer->CreateInfo.BlockSize = BlockSize;
		outBuffer->CreateInfo.AllocationSize = AllocationSize;
		outBuffer->CreateInfo.Offset = Offset;
		outBuffer->CreateInfo.IsImported = true;
		outBuffer->CreateInfo.ImportedInternalHandle = reinterpret_cast<uint64_t>(externalMemory);
		outBuffer->CreateInfo.ImportedExternalHandle = Handle;
		outBuffer->ShareInfo.CreateHandle = NULL;
		outBuffer->ShareInfo.ShareableHandle = NULL;
		outBuffer->MemoryType = MEMORY_TYPE_DEVICE;
		ResManager.Add(outBuffer->Address, *outBuffer);
		return NOS_RESULT_SUCCESS;
	}
	nosResult ImportExternalSemaphore(uint64_t handle, nosCUDAExternalSemaphoreHandleType handleType, nosCUDAExtSemaphore* extSem)
	{
		CHECK_CONTEXT_SWITCH();

		cudaExternalSemaphore_t extSemCuda = NULL;
		cudaExternalSemaphoreHandleDesc desc = {};
		memset(&desc, 0, sizeof(desc));
		desc.type = (cudaExternalSemaphoreHandleType)handleType;
		switch (handleType) {
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEFD:
					desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
					desc.handle.fd = handle;
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEWIN32:
					desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUEWIN32KMT:
					desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12FENCE:
					desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11FENCE:
					desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC:
					desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
					desc.handle.nvSciSyncObj = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_KEYEDMUTEX:
					desc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_KEYEDMUTEXKMT:
					desc.type = cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINESEMAPHOREFD:
					desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
					desc.handle.fd = handle;
					break; 
				case EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINESEMAPHOREWIN32:
					desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
					desc.handle.win32.handle = reinterpret_cast<void*>(handle);
					break;
				default:
					return NOS_RESULT_FAILED;
		}
		cudaError res = cudaImportExternalSemaphore(&extSemCuda, &desc);
		CHECK_CUDA_RT_ERROR(res);
		(*extSem) = extSemCuda;
		return NOS_RESULT_SUCCESS;
	}

	nosResult NOSAPI_CALL DestroyBuffer(nosCUDABufferInfo* cudaBuffer)
	{
		CHECK_CONTEXT_SWITCH();

		CHECK_VALID_ARGUMENT(cudaBuffer);
		CUresult driverRes = CUDA_SUCCESS;
		cudaError rtRes = cudaSuccess;
		if(cudaBuffer->Address != NULL){
			if (!cudaBuffer->CreateInfo.IsImported) {
				if (cudaBuffer->ShareInfo.CreateHandle != NULL) {
					driverRes = cuMemUnmap(cudaBuffer->Address, cudaBuffer->CreateInfo.AllocationSize);
					CHECK_CUDA_DRIVER_ERROR(driverRes);
					driverRes = cuMemAddressFree(cudaBuffer->Address, cudaBuffer->CreateInfo.AllocationSize);
					CHECK_CUDA_DRIVER_ERROR(driverRes);
					driverRes = cuMemRelease(cudaBuffer->ShareInfo.CreateHandle);
					CHECK_CUDA_DRIVER_ERROR(driverRes);
				}
				else {
					rtRes = cudaFree(reinterpret_cast<void*>(cudaBuffer->Address));
					CHECK_CUDA_RT_ERROR(rtRes);
				}
			}
			else {
				rtRes = cudaFree(reinterpret_cast<void*>(cudaBuffer->Address));
				CHECK_CUDA_RT_ERROR(rtRes);

				rtRes = cudaDestroyExternalMemory(reinterpret_cast<cudaExternalMemory_t>(cudaBuffer->CreateInfo.ImportedInternalHandle));
				CHECK_CUDA_RT_ERROR(rtRes);
			}
		}
		
		memset(cudaBuffer, 0, sizeof(nosCUDABufferInfo));//Let the users know

		return NOS_RESULT_SUCCESS;
	}

	FORCEINLINE nosResult ContextSwitch()
	{
		uint64_t ID = NULL;
		nosResult nosRes = GetCallingModuleID(&ID);
		if (nosRes != NOS_RESULT_SUCCESS)
			return nosRes;

		CUresult cuRes = CUDA_SUCCESS;
		if (IDContextMap.contains(ID))
		{
			ActiveContext = IDContextMap[ID];
			cuRes = cuCtxSetCurrent(reinterpret_cast<CUcontext>(ActiveContext));
			CHECK_CUDA_DRIVER_ERROR(cuRes);
		}
		else if(ActiveContext != PrimaryContext)
		{
			ActiveContext = PrimaryContext;
			cuRes = cuCtxSetCurrent(reinterpret_cast<CUcontext>(ActiveContext));
			CHECK_CUDA_DRIVER_ERROR(cuRes);
		}
		return NOS_RESULT_SUCCESS;
	}

	FORCEINLINE nosResult GetCallingModuleID(uint64_t* ID)
	{
		nosModuleContext moduleContext = {};
		nosResult nosRes = nosEngine.GetCallingModule(&moduleContext);
		if (nosRes != NOS_RESULT_SUCCESS)
			return nosRes;
		(*ID) = moduleContext.Id.Name;
		return NOS_RESULT_SUCCESS;
	}
}
