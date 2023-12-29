#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

extern "C" {
	void NormalizeKernelWrapper(dim3 blocks, dim3 thread, float* data, float max, int dataSize);
}