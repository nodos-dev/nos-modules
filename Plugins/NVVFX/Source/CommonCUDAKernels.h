#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

extern "C" {
	void NormalizeKernelWrapper(dim3 blocks, dim3 thread, int* data, int max, int dataSize);
}