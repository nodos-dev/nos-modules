#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "nvCVImage.h"
#include <Nodos/PluginHelpers.hpp>
#include "cuda.h"

#include <Windows.h>

class CudaGPUResourceManager {
public:
	CudaGPUResourceManager();
	~CudaGPUResourceManager();

	nosResult InitializeCUDADevice(int device = 0);
	int QueryCudaDeviceCount();
	int64_t AllocateGPU(std::string name, size_t count);

	CUmemGenericAllocationHandle AllocateShareableGPU(std::string name, size_t count);

	int64_t GetGPUBuffer(std::string name);
	nosResult MemCopy(std::string source, std::string destination);
	nosResult MemCopy(int64_t source, std::string destination, int64_t size = 0);
	nosResult MemCopy(std::string source, int64_t destination);
	nosResult MemCopy(int64_t source, int64_t destination, int64_t size);

	int64_t GetSize(std::string name);

private:
	NvCVImage NVVFX_Image;
	std::unordered_map<std::string, int64_t> GPUBufferAddresses;
	std::unordered_map<std::string, int64_t> GPUBufferSizes;
	std::unordered_map<std::string, int64_t> GPUShareableAddresses;

};
