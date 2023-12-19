#include "CUDAResourceManager.h"

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

template <typename T>
void* CudaGPUResourceManager::AllocateGPU(std::string name, size_t count)
{
	T* def = nullptr;
	cudaError_t res = cudaMalloc((void**)&def, count);
	if (res != cudaError::cudaSuccess)
		return nullptr;
	GPUBufferAddresses.emplace(std::move(name), def);
	return def;
}

void* CudaGPUResourceManager::GetGPUBuffer(std::string name)
{
	if (GPUBufferAddresses.contains(name)) {
		return GPUBufferAddresses[name];
	}
	return nullptr;
}
