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
    DisposeResources();
}

void CudaGPUResourceManager::DisposeResources()
{
    for (const auto [name, _] : CUDABuffers) {
        FreeGPUBuffer(name);
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
	uint64_t def = NULL;;
	cudaError_t res = cudaMalloc((void**)&def, count);
	if (res != cudaError::cudaSuccess)
		return NULL;

    CUDABuffers[name] = { .address = def, .size = count };

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

    

    uint64_t shareableHandle = 0;
    status = cuMemExportToShareableHandle((void *)&shareableHandle, handle, CU_MEM_HANDLE_TYPE_WIN32, 0);
    const char* errorMsg;
    cuGetErrorString(status, &errorMsg);
    assert(status == CUDA_SUCCESS);

    CUDABuffers[name] = {.address = new_ptr, .shareableHandle = shareableHandle, .size = aligned_size };
    return shareableHandle;
}

nosResult CudaGPUResourceManager::GetGPUBuffer(std::string name, uint64_t* buffer)
{
	if (CUDABuffers.contains(name)) {
        (*buffer) = CUDABuffers[name].address;
        return NOS_RESULT_SUCCESS;
	}
    return NOS_RESULT_FAILED;
}

nosResult CudaGPUResourceManager::GetShareableHandle(uint64_t bufferAddress, uint64_t* shareableHandle)
{
    for (const auto [name, buff] : CUDABuffers) {
        if (buff.address == bufferAddress) {
            (*shareableHandle) = buff.shareableHandle;
            return NOS_RESULT_SUCCESS;
        }
    }
    return NOS_RESULT_FAILED;
}

nosResult CudaGPUResourceManager::GetShareableHandle(std::string name, uint64_t* shareableHandle)
{
    if (CUDABuffers.contains(name)) {
        (*shareableHandle) = CUDABuffers[name].shareableHandle;
        return NOS_RESULT_SUCCESS;
    }
    return NOS_RESULT_FAILED;
}

nosResult CudaGPUResourceManager::MemCopy(std::string source, std::string destination, int64_t size)
{
    if (CUDABuffers[source].size != CUDABuffers[destination].size) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }
    cudaError res = cudaMemcpy((void*)&CUDABuffers[destination].address, (void*)&CUDABuffers[source].address, CUDABuffers[source].size, cudaMemcpyDeviceToDevice);
    assert(res == CUDA_SUCCESS);

    return NOS_RESULT_SUCCESS;
}

nosResult CudaGPUResourceManager::MemCopy(int64_t source, std::string destination, int64_t size)
{
    if (!CUDABuffers.contains(destination)) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    cudaError res = cudaMemcpy(reinterpret_cast<void*>(CUDABuffers[destination].address), reinterpret_cast<void*>(source), CUDABuffers[destination].size, cudaMemcpyDeviceToDevice);
    assert(res == CUDA_SUCCESS);

    return NOS_RESULT_SUCCESS;
}

nosResult CudaGPUResourceManager::MemCopy(std::string source, int64_t destination)
{
    if (!CUDABuffers.contains(source)) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    cudaError res = cudaMemcpy(reinterpret_cast<void*>(destination), reinterpret_cast<void*>(CUDABuffers[source].address), CUDABuffers[source].size, cudaMemcpyDeviceToDevice);
    return NOS_RESULT_SUCCESS;
}

nosResult CudaGPUResourceManager::MemCopy(int64_t source, int64_t destination, int64_t size)
{
    if (size <= 0) {
        nosEngine.LogE("Buffer size mismatch");
        return NOS_RESULT_FAILED;
    }

    uint64_t internalAddres_src = NULL;
    uint64_t internalAddres_dst = NULL;

    for (const auto& [_name, _buf] : CUDABuffers) {
        if (_buf.shareableHandle == source) {
            internalAddres_src = _buf.address;
        }
        if (_buf.shareableHandle == destination) {
            internalAddres_dst = _buf.address;
        }
    }

    internalAddres_dst = (internalAddres_dst == NULL) ? (destination) : (internalAddres_dst);
    internalAddres_src = (internalAddres_src == NULL) ? (source) : (internalAddres_src);

    cudaError res = cudaMemcpy(reinterpret_cast<void*>(internalAddres_dst), reinterpret_cast<void*>(internalAddres_src), size, cudaMemcpyDeviceToDevice);
    cudaError syncRes = cudaDeviceSynchronize();
    if (syncRes != CUDA_SUCCESS) {
        nosEngine.LogE("CUDA device synchronize failed with error code %d", syncRes);
        return NOS_RESULT_FAILED;
    }

    assert(res == cudaSuccess);

    //uint8_t* src = new uint8_t[size];
    //cudaMemcpy(src, reinterpret_cast<void*>(internalAddres_src), size, cudaMemcpyDeviceToHost);

    //uint8_t* dst = new uint8_t[size];
    //cudaMemcpy(dst, reinterpret_cast<void*>(internalAddres_dst), size, cudaMemcpyDeviceToHost);


    if (res != cudaSuccess) {
        nosEngine.LogE("CudaMemcyp failed with error code %d!", res);
        return NOS_RESULT_FAILED;
    }
    //delete[] src;
    //delete[] dst;
    return NOS_RESULT_SUCCESS;
}

int64_t CudaGPUResourceManager::GetSize(std::string name)
{
    if (!CUDABuffers.contains(name))
        return 0;

    return CUDABuffers[name].size;
}

nosResult CudaGPUResourceManager::FreeGPUBuffer(std::string name)
{
    assert(CUDABuffers.contains(name));
    if (!CUDABuffers.contains(name))
        return NOS_RESULT_FAILED;

    CUresult res = CUDA_SUCCESS;
    cudaError cudaRes = cudaSuccess;
    auto buffer = CUDABuffers[name];
    if (buffer.shareableHandle == NULL) {
        cudaRes = cudaFree((void*)&buffer.address);
    }
    else {
        res = cuMemUnmap(buffer.address, buffer.size);
        res = cuMemAddressFree(buffer.address, buffer.size);
        res = cuMemRelease(buffer.shareableHandle);
    }

    if (res != CUDA_SUCCESS || cudaRes != cudaSuccess)
        return NOS_RESULT_FAILED;

    return NOS_RESULT_SUCCESS;
}

