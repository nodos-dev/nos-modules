#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "nvCVImage.h"
#include <Nodos/PluginHelpers.hpp>

class CudaGPUResourceManager {
public:
	CudaGPUResourceManager();
	~CudaGPUResourceManager();

	nosResult InitializeCUDADevice(int device = 0);
	int QueryCudaDeviceCount();
	template <typename T>
	void* AllocateGPU(std::string name, size_t count);
	void* GetGPUBuffer(std::string name);
private:
	NvCVImage NVVFX_Image;
	std::unordered_map<std::string,void*> GPUBufferAddresses;

};
