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
	int* AllocateGPU(std::string name, size_t count);
private:
	NvCVImage NVVFX_Image;
	std::unordered_map<std::string,int*> GPUBufferAddresses;

};
