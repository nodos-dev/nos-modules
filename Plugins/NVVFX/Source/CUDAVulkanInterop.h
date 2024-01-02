#pragma once
#include <Nodos/PluginHelpers.hpp>
#include <cuda_runtime.h>
#include "cuda.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"
#include "nosVulkanSubsystem/Types_generated.h"
#include "nvCVImage.h"
#include "CUDAResourceManager.h"

enum nosNVCVLayout {
	nosNVCV_INTERLEAVED   =	0,//!< All components of pixel(x,y) are adjacent (same as chunky) (default for non-YUV).
	nosNVCV_CHUNKY        =	0,//!< All components of pixel(x,y) are adjacent (same as interleaved).
	nosNVCV_PLANAR        =	1,//!< The same component of all pixels are adjacent.
	nosNVCV_UYVY          =	2,//!< [UYVY]    Chunky 4:2:2 (default for 4:2:2)
	nosNVCV_VYUY          =	4,//!< [VYUY]    Chunky 4:2:2
	nosNVCV_YUYV          =	6,//!< [YUYV]    Chunky 4:2:2
	nosNVCV_YVYU          =	8,//!< [YVYU]    Chunky 4:2:2
	nosNVCV_CYUV         =	10,//!< [YUV]     Chunky 4:4:4
	nosNVCV_CYVU         =	12,//!< [YVU]     Chunky 4:4:4
	nosNVCV_YUV           =	3,//!< [Y][U][V] Planar 4:2:2 or 4:2:0 or 4:4:4
	nosNVCV_YVU           =	5,//!< [Y][V][U] Planar 4:2:2 or 4:2:0 or 4:4:4
	nosNVCV_YCUV          =	7,//!< [Y][UV]   Semi-planar 4:2:2 or 4:2:0 (default for 4:2:0)
	nosNVCV_YCVU          =	9,//!< [Y][VU]   Semi-planar 4:2:2 or 4:2:0
};

class CUDAVulkanInterop {
public:
	CUDAVulkanInterop();
	~CUDAVulkanInterop();

	nosResult SetVulkanMemoryToCUDA(int64_t handle, size_t size, size_t offset, uint64_t* outCudaPointerAddres);

	nosResult nosTextureToNVCVImage(nosResourceShareInfo& vulkanTex, NvCVImage& nvcvImage, std::optional<nosNVCVLayout> layout = std::nullopt);
	nosResult NVCVImageToNosTexture(NvCVImage& nvcvImage, nosResourceShareInfo& vulkanTex, std::optional<nosNVCVLayout> layout = std::nullopt);

	//User must ensure the correctness of FORMAT, width, height of the required texture. CUDA will only set GPU addresses
	void SetCUDAMemoryToVulkan(int64_t cudaPointerAddress, int width, int height, size_t size, size_t offset, int32_t format, nos::sys::vulkan::TTexture* outNosTexture);
	nosResult AllocateNVCVImage(std::string name, int width, int height, NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType, size_t size ,NvCVImage* out);
	
	NvCVImage_PixelFormat GetPixelFormatFromVulkanFormat(nosFormat format);
	NvCVImage_ComponentType GetComponentTypeFromVulkanFormat(nosFormat format);
	nosFormat GetVulkanFormatFromNVCVImage(NvCVImage nvcvImage);
	void NormalizeNVCVImage(NvCVImage* nvcvImage);
	nosResult CopyNVCVImage(NvCVImage* dst, NvCVImage* src);
private:
	void InitCUDA();
	CudaGPUResourceManager GPUResManager;
};