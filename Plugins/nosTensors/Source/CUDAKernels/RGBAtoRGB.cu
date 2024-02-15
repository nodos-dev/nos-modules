#include "cuda.h"
#include "device_launch_parameters.h"
extern "C" {
	__global__ void RGBAtoRGB(void* InData, int TotalSizeInBytes, int BytesPerElement, int Width, int Height, void* OutData) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		char* inData = reinterpret_cast<char*>(InData);
		char* outData = reinterpret_cast<char*>(OutData);
		idx = idx * BytesPerElement;
		if ((idx + Width*Height*2*BytesPerElement + BytesPerElement) < (TotalSizeInBytes) && (idx*4 + 3*BytesPerElement) < TotalSizeInBytes) {
			for (int i = 0; i < BytesPerElement; i++) { //Copy all bytes!
								   				   *(outData + idx + i)	= *(inData + i + idx*4);				 	 //R
				   *(outData + (idx + Width*Height*BytesPerElement)+ i) = *(inData + i + idx*4 + 1*BytesPerElement); //G
				 *(outData + (idx + Width*Height*2*BytesPerElement)+ i)	= *(inData + i + idx*4 + 2*BytesPerElement); //B
			}
		}
	}
}