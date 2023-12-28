#include "CUDAResourceManager.h"

namespace security {
    #include "ntdef.h"
    #include "sddl.h"
    
    static void
    getDefaultSecurityDescriptor(CUmemAllocationProp* prop)
    {
    #if defined(__linux__)
        return;
    #elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
        static OBJECT_ATTRIBUTES objAttributes;
        static bool objAttributesConfigured = false;
    
        if (!objAttributesConfigured) {
            PSECURITY_DESCRIPTOR secDesc;
            BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1, &secDesc, NULL);
            if (result == 0) {
                printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n", GetLastError());
            }
    
            InitializeObjectAttributes(
                &objAttributes,
                NULL,
                0,
                NULL,
                secDesc
            );
    
            objAttributesConfigured = true;
        }
    
        prop->win32HandleMetaData = &objAttributes;
        return;
    #endif
    }
}

CudaGPUResourceManager::CudaGPUResourceManager()
{
}

CudaGPUResourceManager::~CudaGPUResourceManager()
{
	for (const auto& [_, ptr] : GPUBufferAddresses) {
		//TODO: check the result!
		cudaFree((void*)&ptr);
	}
}

nosResult CudaGPUResourceManager::InitializeCUDADevice(int device)
{
	cudaError_t res = cudaSetDevice(device);

	if (res == cudaError::cudaSuccess)
		return NOS_RESULT_SUCCESS;

	return NOS_RESULT_FAILED;
}

int CudaGPUResourceManager::QueryCudaDeviceCount()
{
	int count;
	cudaError_t res = cudaGetDeviceCount(&count);
	if (res != cudaError::cudaSuccess)
		return -1;
	return count;
}

int64_t CudaGPUResourceManager::AllocateGPU(std::string name, size_t count)
{
	int64_t def = NULL;;
	cudaError_t res = cudaMalloc((void**)&def, count);
	if (res != cudaError::cudaSuccess)
		return NULL;

    GPUBufferSizes.emplace(name, count);
    GPUBufferAddresses.emplace(std::move(name), def);

	return def;
}

CUmemGenericAllocationHandle CudaGPUResourceManager::AllocateShareableGPU(std::string name, size_t size)
{
    

    CUcontext ctx;
    CUdevice dev;
    int supportsVMM = 0, supportsWin32 = 0;
    
    cuDevicePrimaryCtxRetain(&ctx, 0);
    cuCtxSetCurrent(ctx);
    cuCtxGetDevice(&dev);

    cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
    assert(supportsVMM == 1);

    cuDeviceGetAttribute(&supportsWin32,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, dev);
    assert(supportsWin32 == 1);

    CUresult status = CUDA_SUCCESS;
    CUmemAllocationProp prop;
    
    memset(&prop, 0, sizeof(prop));
    SECURITY_ATTRIBUTES lps = {};
    lps.nLength = sizeof(SECURITY_ATTRIBUTES);
    lps.lpSecurityDescriptor = NULL;  // Use a NULL security descriptor for default security settings.
    lps.bInheritHandle = TRUE;       // Set to TRUE if you want the handle to be inheritable.


    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)dev;
    prop.win32HandleMetaData = &lps;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    security::getDefaultSecurityDescriptor(&prop);

    size_t chunk_sz;
    status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);
    const size_t aligned_size = ((size + chunk_sz - 1) / chunk_sz) * chunk_sz;

    CUmemGenericAllocationHandle handle;

    status = cuMemCreate(&handle, aligned_size, &prop, 0);
    assert(status == CUDA_SUCCESS);

    CUdeviceptr new_ptr = 0ULL;
    status = cuMemAddressReserve(&new_ptr, (aligned_size), 0ULL, 0ULL, 0ULL);
    assert(status == CUDA_SUCCESS);

    status = cuMemMap(new_ptr, aligned_size, 0, handle, 0);
    assert(status == CUDA_SUCCESS);

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = dev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // Make the address accessible
    status = cuMemSetAccess(new_ptr, aligned_size, &accessDesc, 1);
    assert(status == CUDA_SUCCESS);

    int64_t shareableHandle = 0;
    status = cuMemExportToShareableHandle((void *)&shareableHandle, handle, CU_MEM_HANDLE_TYPE_WIN32, 0);
    const char* errorMsg;
    cuGetErrorString(status, &errorMsg);
    assert(status == CUDA_SUCCESS);

    GPUBufferAddresses[name] = new_ptr;
    GPUBufferSizes[name] = aligned_size;
	GPUShareableAddresses[name] = shareableHandle;
	return shareableHandle;
}

int64_t CudaGPUResourceManager::GetGPUBuffer(std::string name)
{
	if (GPUBufferAddresses.contains(name)) {
		return GPUBufferAddresses[name];
	}
    return NULL;
}

nosResult CudaGPUResourceManager::MemCopy(std::string source, std::string destination)
{
    if (GPUBufferSizes[source] != GPUBufferSizes[destination]) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }
    cudaMemcpy((void*)&GPUBufferAddresses[destination], (void*)&GPUBufferAddresses[source], GPUBufferSizes[source], cudaMemcpyDeviceToDevice);
    return nosResult();
}

nosResult CudaGPUResourceManager::MemCopy(int64_t source, std::string destination, int64_t size)
{
    if (!GPUBufferSizes.contains(destination)) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    cudaError_t res = cudaMemset(reinterpret_cast<void*>(GPUBufferAddresses[destination]), 255, size);
    assert(res == CUDA_SUCCESS);

    return NOS_RESULT_SUCCESS;
}

nosResult CudaGPUResourceManager::MemCopy(std::string source, int64_t destination)
{
    if (!GPUBufferSizes.contains(source)) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    cudaMemcpy(reinterpret_cast<void*>(destination), reinterpret_cast<void*>(GPUBufferAddresses[source]), GPUBufferSizes[source], cudaMemcpyDeviceToDevice);
    return NOS_RESULT_SUCCESS;
}

nosResult CudaGPUResourceManager::MemCopy(int64_t source, int64_t destination, int64_t size)
{
    if (size <= 0) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    cudaMemcpy((void*)&destination, (void*)&source, size, cudaMemcpyDeviceToDevice);
    return NOS_RESULT_SUCCESS;
}

int64_t CudaGPUResourceManager::GetSize(std::string name)
{
    if (!GPUBufferSizes.contains(name))
        return 0;

    return GPUBufferSizes[name];
}

nosResult CudaGPUResourceManager::FreeGPUBuffer()
{
    
    return nosResult();
}

