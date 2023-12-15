#include "NVVFXCudaVulkanInterop.h"

CudaGPUResourceManager::CudaGPUResourceManager()
{
}

CudaGPUResourceManager::~CudaGPUResourceManager()
{
	for (const auto& [_, ptr] : GPUBufferAddresses) {
		//TODO: check the result!
		cudaFree(ptr);
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

int* CudaGPUResourceManager::AllocateGPU(std::string name, size_t count)
{
	int* def = nullptr;
	cudaError_t res = cudaMalloc((void**)&def, count);
	if (res != cudaError::cudaSuccess)
		return nullptr;
	GPUBufferAddresses.emplace(std::move(name), def);
	return def;
}
