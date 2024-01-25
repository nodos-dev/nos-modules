#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

__global__ void NormalizeKernel(float* data, float max, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        data[idx] = (data[idx])/max;
    }
}