#include "cuda.h"
#include "device_launch_parameters.h"
extern "C" {
__global__ void SliceTensor(void* InData, int DataSize, int ByteStride, int ElementSize, void* OutData) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	char* inData = reinterpret_cast<char*>(InData);
	char* outData = reinterpret_cast<char*>(OutData);
	idx = idx * ByteStride;
	if (idx < DataSize) {
		for (int i = 0; i < ElementSize; i++) { //Copy all bytes!
			*(outData + idx + i) = *(inData + i);
		}
	}
}
}