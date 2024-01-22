#pragma once
#include "nosCommon.h"
#include "nosDefines.h"
#include <Nodos/SubsystemAPI.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda.h"

#include <Windows.h>

#define CHECK_CUDA_RT_ERROR(cudaRes)	\
	do{							\
		if (cudaRes != cudaSuccess) {	\
			nosEngine.LogE("CUDA RT failed with error: %s", cudaGetErrorString(cudaRes));	\
			return NOS_RESULT_FAILED; \
		}						\
	}while(0)					\

struct CUDABuffer {
	uint64_t address = NULL;
	uint64_t shareableHandle = NULL;
	uint64_t size = 0;
	CUmemGenericAllocationHandle createHandle = NULL;
};

class CudaGPUResourceManager {
public:
	CudaGPUResourceManager();
	~CudaGPUResourceManager();
	nosResult FreeGPUBuffer(std::string name);
	void DisposeAllResources();

	nosResult InitializeCUDADevice(int device = 0);
	int QueryCudaDeviceCount();
	int64_t AllocateGPU(std::string name, size_t count);

	CUmemGenericAllocationHandle AllocateShareableGPU(std::string name, size_t count);

	nosResult GetGPUBuffer(std::string name, uint64_t* buffer);
	nosResult GetShareableHandle(uint64_t bufferAddress, uint64_t* shareableHandle);
	nosResult GetShareableHandle(std::string name, uint64_t* shareableHandle);

	//TODO: get rid of this boilerplate stuff -> forward overloaded functions to single pure one
	nosResult MemCopy(std::string source, std::string destination, int64_t size);
	nosResult MemCopy(int64_t source, std::string destination, int64_t size);
	nosResult MemCopy(std::string source, int64_t destination);
	nosResult MemCopy(int64_t source, int64_t destination, int64_t size);

	int64_t GetSize(std::string name);

	bool IsResourceExist(std::string name);

private:
	
	std::unordered_map<std::string, CUDABuffer> CUDABuffers;

};
