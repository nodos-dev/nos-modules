#include "CUDAVulkanInterop.h"
#include "Nodos/Helpers.hpp"
#include "nosVulkanSubsystem/Helpers.hpp"
/*
typedef struct nosMemoryInfo
{
	uint32_t Type;
	uint64_t Handle;
	uint64_t PID;
	uint64_t Memory;
	uint64_t Offset;
} nosMemoryInfo;
*/
nosResult CUDAVulkanInterop::SetVulkanMemoryToCUDA(int64_t handle, size_t size, size_t offset, int64_t& outCudaPointerAddres)
{
	outCudaPointerAddres = NULL;
	CUexternalMemory externalMemory;
	CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc;
	desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
	desc.handle.win32.handle = reinterpret_cast<void*>(handle);
	desc.handle.win32.name = NULL;
	desc.size = size;
	
	CUresult res = cuImportExternalMemory(&externalMemory, &desc);

	if (res != CUresult::CUDA_SUCCESS) {
		nosEngine.LogE("Import attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}

	CUdeviceptr pointer;

	CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc;
	bufferDesc.flags = 0; // must be zero
	bufferDesc.offset = offset;
	bufferDesc.size = size;

	res = cuExternalMemoryGetMappedBuffer(&pointer, externalMemory, &bufferDesc);
	if (res != CUresult::CUDA_SUCCESS || pointer == NULL) {
		nosEngine.LogE("Get mapped buffer attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}
	
	outCudaPointerAddres = pointer;
	return NOS_RESULT_SUCCESS;
}

void CUDAVulkanInterop::SetCUDAMemoryToVulkan(int64_t cudaPointerAddress, int width, int height ,size_t size, size_t offset, int32_t format, nos::sys::vulkan::TTexture& outNosTexture) {
	outNosTexture.size = nos::sys::vulkan::SizePreset::CUSTOM;
	outNosTexture.width = width;
	outNosTexture.height = height;
	outNosTexture.format = nos::sys::vulkan::Format(format);
	outNosTexture.usage |= nos::sys::vulkan::ImageUsage::SAMPLED;
	outNosTexture.type = 0x00000002;
	outNosTexture.memory = cudaPointerAddress;
	outNosTexture.pid = getpid();
	outNosTexture.unmanaged = true;
	outNosTexture.unscaled = true;
	outNosTexture.offset = offset;
	outNosTexture.handle = 0;
	outNosTexture.semaphore = 0;
}

nosResult CUDAVulkanInterop::nosTextureToNVCVImage(nosResourceShareInfo& vulkanTex, NvCVImage& nvcvImage, std::optional<nosNVCVLayout> layout)
{
	int64_t gpuPointer = NULL;
	size_t textureSize = vulkanTex.Info.Texture.GetSize();
	nosResult res = SetVulkanMemoryToCUDA(vulkanTex.Memory.Memory, textureSize, vulkanTex.Memory.Offset, gpuPointer);
	if (res != NOS_RESULT_SUCCESS)
		return res;

	nvcvImage.pixels = reinterpret_cast<void*>(gpuPointer);
	nvcvImage.width = vulkanTex.Info.Texture.Width;
	nvcvImage.height = vulkanTex.Info.Texture.Height;
	nvcvImage.gpuMem = NVCV_CUDA;
	nvcvImage.pitch = textureSize / vulkanTex.Info.Texture.Height;
	nvcvImage.bufferBytes = textureSize;
	nvcvImage.pixelFormat = GetPixelFormatFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.componentType = GetComponentTypeFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.planar = (layout == std::nullopt) ? (nvcvImage.planar) : ((int)layout.value());

	return res;
}

nosResult CUDAVulkanInterop::NVCVImageToNosTexture(NvCVImage& nvcvImage, nosResourceShareInfo& vulkanTex, std::optional<nosNVCVLayout> layout)
{
	auto tex = nos::vkss::ConvertTextureInfo(vulkanTex);
	
	SetCUDAMemoryToVulkan(reinterpret_cast<uint64_t>(nvcvImage.pixels), nvcvImage.width, nvcvImage.height, 
		vulkanTex.Info.Texture.GetSize(), vulkanTex.Memory.Offset, GetVulkanFormatFromNVCVImage(nvcvImage), tex);
	
	vulkanTex = nos::vkss::DeserializeTextureInfo(nos::Buffer::From(tex)); //may be a better way??

	return NOS_RESULT_SUCCESS;
}

NvCVImage_PixelFormat CUDAVulkanInterop::GetPixelFormatFromVulkanFormat(nosFormat format)
{
	switch (format) {
			case NOS_FORMAT_NONE:
			case NOS_FORMAT_R8_UNORM:
			case NOS_FORMAT_R8_UINT:
			case NOS_FORMAT_R8_SRGB:
				return NvCVImage_PixelFormat::NVCV_A;
			case NOS_FORMAT_R8G8_UNORM:
			case NOS_FORMAT_R8G8_UINT:
			case NOS_FORMAT_R8G8_SRGB:
				return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
			case NOS_FORMAT_R8G8B8_UNORM:
			case NOS_FORMAT_R8G8B8_SRGB:
				return NvCVImage_PixelFormat::NVCV_RGB;
			case NOS_FORMAT_B8G8R8_UNORM:
			case NOS_FORMAT_B8G8R8_UINT:
			case NOS_FORMAT_B8G8R8_SRGB:
				return NvCVImage_PixelFormat::NVCV_BGR;
			case NOS_FORMAT_R8G8B8A8_UNORM:
			case NOS_FORMAT_R8G8B8A8_UINT:
			case NOS_FORMAT_R8G8B8A8_SRGB:
				return NvCVImage_PixelFormat::NVCV_RGBA;
			case NOS_FORMAT_B8G8R8A8_UNORM:
			case NOS_FORMAT_B8G8R8A8_SRGB:
				return NvCVImage_PixelFormat::NVCV_BGRA;
			case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
			case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
			case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
			case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
			case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
			case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
				return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
			case NOS_FORMAT_R16_UNORM:
			case NOS_FORMAT_R16_SNORM:
			case NOS_FORMAT_R16_USCALED:
			case NOS_FORMAT_R16_SSCALED:
			case NOS_FORMAT_R16_UINT:
			case NOS_FORMAT_R16_SINT:
			case NOS_FORMAT_R16_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_A;
			case NOS_FORMAT_R16G16_UNORM:
			case NOS_FORMAT_R16G16_SNORM:
			case NOS_FORMAT_R16G16_USCALED:
			case NOS_FORMAT_R16G16_SSCALED:
			case NOS_FORMAT_R16G16_UINT:
			case NOS_FORMAT_R16G16_SINT:
			case NOS_FORMAT_R16G16_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
			case NOS_FORMAT_R16G16B16_UNORM:
			case NOS_FORMAT_R16G16B16_SNORM:
			case NOS_FORMAT_R16G16B16_USCALED:
			case NOS_FORMAT_R16G16B16_SSCALED:
			case NOS_FORMAT_R16G16B16_UINT:
			case NOS_FORMAT_R16G16B16_SINT:
			case NOS_FORMAT_R16G16B16_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_RGB;
			case NOS_FORMAT_R16G16B16A16_UNORM:
			case NOS_FORMAT_R16G16B16A16_SNORM:
			case NOS_FORMAT_R16G16B16A16_USCALED:
			case NOS_FORMAT_R16G16B16A16_SSCALED:
			case NOS_FORMAT_R16G16B16A16_UINT:
			case NOS_FORMAT_R16G16B16A16_SINT:
			case NOS_FORMAT_R16G16B16A16_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_RGBA;
			case NOS_FORMAT_R32_UINT:
			case NOS_FORMAT_R32_SINT:
			case NOS_FORMAT_R32_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_A;
			case NOS_FORMAT_R32G32_UINT:
			case NOS_FORMAT_R32G32_SINT:
			case NOS_FORMAT_R32G32_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
			case NOS_FORMAT_R32G32B32_UINT:
			case NOS_FORMAT_R32G32B32_SINT:
			case NOS_FORMAT_R32G32B32_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_RGB;
			case NOS_FORMAT_R32G32B32A32_UINT:
			case NOS_FORMAT_R32G32B32A32_SINT:
			case NOS_FORMAT_R32G32B32A32_SFLOAT:
				return NvCVImage_PixelFormat::NVCV_RGBA;
			case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
			case NOS_FORMAT_D16_UNORM:
			case NOS_FORMAT_X8_D24_UNORM_PACK32:
			case NOS_FORMAT_D32_SFLOAT:
			case NOS_FORMAT_G8B8G8R8_422_UNORM:
			case NOS_FORMAT_B8G8R8G8_422_UNORM:
				return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
	}
	return NvCVImage_PixelFormat::NVCV_FORMAT_UNKNOWN;
}

NvCVImage_ComponentType CUDAVulkanInterop::GetComponentTypeFromVulkanFormat(nosFormat format) {
	switch (format) {
		case NOS_FORMAT_R8_UNORM:
		case NOS_FORMAT_R8G8_UNORM:
		case NOS_FORMAT_R8G8B8_UNORM:
		case NOS_FORMAT_B8G8R8_UNORM:
		case NOS_FORMAT_R8G8B8A8_UNORM:
		case NOS_FORMAT_B8G8R8A8_UNORM:
		case NOS_FORMAT_G8B8G8R8_422_UNORM:
		case NOS_FORMAT_B8G8R8G8_422_UNORM:
		case NOS_FORMAT_R8_UINT:
		case NOS_FORMAT_R8G8_UINT:
		case NOS_FORMAT_B8G8R8_UINT:
		case NOS_FORMAT_R8G8B8A8_UINT:
		case NOS_FORMAT_R8_SRGB:
		case NOS_FORMAT_R8G8_SRGB:
		case NOS_FORMAT_R8G8B8_SRGB:
		case NOS_FORMAT_B8G8R8_SRGB:
		case NOS_FORMAT_R8G8B8A8_SRGB:
		case NOS_FORMAT_B8G8R8A8_SRGB:
			return NvCVImage_ComponentType::NVCV_U8;
		
		case NOS_FORMAT_R16_UNORM:
		case NOS_FORMAT_R16G16_UNORM:
		case NOS_FORMAT_R16G16B16_UNORM:
		case NOS_FORMAT_R16G16B16A16_UNORM:
		case NOS_FORMAT_D16_UNORM:
		case NOS_FORMAT_R16_UINT:
		case NOS_FORMAT_R16G16B16_UINT:
		case NOS_FORMAT_R16G16_UINT:
		case NOS_FORMAT_R16G16B16A16_UINT:
		case NOS_FORMAT_R32_UINT:
		case NOS_FORMAT_R16_USCALED:
		case NOS_FORMAT_R16G16_USCALED:
		case NOS_FORMAT_R16G16B16_USCALED:
		case NOS_FORMAT_R16G16B16A16_USCALED:
			return NvCVImage_ComponentType::NVCV_U16;

		case NOS_FORMAT_R32G32_UINT:
		case NOS_FORMAT_R32G32B32_UINT:
		case NOS_FORMAT_R32G32B32A32_UINT:
		case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
		case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
		case NOS_FORMAT_X8_D24_UNORM_PACK32:
			return NvCVImage_ComponentType::NVCV_U32;

		case NOS_FORMAT_R16_SINT:
		case NOS_FORMAT_R16G16_SINT:
		case NOS_FORMAT_R16G16B16_SINT:
		case NOS_FORMAT_R16G16B16A16_SINT:
		case NOS_FORMAT_R16_SNORM:
		case NOS_FORMAT_R16G16_SNORM:
		case NOS_FORMAT_R16G16B16_SNORM:
		case NOS_FORMAT_R16G16B16A16_SNORM:
		case NOS_FORMAT_R16_SSCALED:
		case NOS_FORMAT_R16G16_SSCALED:
		case NOS_FORMAT_R16G16B16_SSCALED:
		case NOS_FORMAT_R16G16B16A16_SSCALED:
			return NvCVImage_ComponentType::NVCV_S16;
		
		case NOS_FORMAT_R16_SFLOAT:
		case NOS_FORMAT_R16G16_SFLOAT:
		case NOS_FORMAT_R16G16B16_SFLOAT:
		case NOS_FORMAT_R16G16B16A16_SFLOAT:
			return NvCVImage_ComponentType::NVCV_F16;

		case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
		case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
		case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
		case NOS_FORMAT_R32_SINT:
		case NOS_FORMAT_R32G32_SINT:
		case NOS_FORMAT_R32G32B32_SINT:
		case NOS_FORMAT_R32G32B32A32_SINT:
			return NvCVImage_ComponentType::NVCV_S32;
		
		case NOS_FORMAT_R32_SFLOAT:
		case NOS_FORMAT_R32G32_SFLOAT:
		case NOS_FORMAT_R32G32B32_SFLOAT:
		case NOS_FORMAT_R32G32B32A32_SFLOAT:
		case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
		case NOS_FORMAT_D32_SFLOAT:
			return NvCVImage_ComponentType::NVCV_F32;
		
		//Cant be mapped for now
		//return NvCVImage_ComponentType::NVCV_U64;
		//return NvCVImage_ComponentType::NVCV_S64;
		//return NvCVImage_ComponentType::NVCV_F64;
		
		}
	return NvCVImage_ComponentType::NVCV_TYPE_UNKNOWN;
}

//A simple trick to handle multi variable switch statement
constexpr uint64_t SwitchPair(NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType) {
	return ((pixelFormat << 16) + compType);
}

nosFormat CUDAVulkanInterop::GetVulkanFormatFromNVCVImage(NvCVImage nvcvImage)
{
	switch (SwitchPair(nvcvImage.pixelFormat, nvcvImage.componentType)) {
		case SwitchPair(NVCV_A, NVCV_U8):
			return NOS_FORMAT_R8_UINT;
		case SwitchPair(NVCV_RGB, NVCV_U8):
			//return NOS_FORMAT_R8G8B8_UINT;
		case SwitchPair(NVCV_BGR, NVCV_U8):
			return NOS_FORMAT_B8G8R8_UINT;
		case SwitchPair(NVCV_RGBA, NVCV_U8):
			return NOS_FORMAT_R8G8B8A8_UINT;
		case SwitchPair(NVCV_BGRA, NVCV_U8):
			//return NOS_FORMAT_B8G8R8A8_UINT;

		case SwitchPair(NVCV_A, NVCV_U16):
			return NOS_FORMAT_R16_UINT;
		case SwitchPair(NVCV_RGB, NVCV_U16):
			return NOS_FORMAT_R16G16B16_UINT;
		case SwitchPair(NVCV_BGR, NVCV_U16):
			//return NOS_FORMAT_B16G16R16_UINT;
		case SwitchPair(NVCV_RGBA, NVCV_U16):
			return NOS_FORMAT_R16G16B16A16_UINT;
		case SwitchPair(NVCV_BGRA, NVCV_U16):
			//return NOS_FORMAT_R16G16B16A16_UINT;

		case SwitchPair(NVCV_A, NVCV_S16):
			return NOS_FORMAT_R16_SINT;
		case SwitchPair(NVCV_RGB, NVCV_S16):
			return NOS_FORMAT_R16G16B16_SINT;
		case SwitchPair(NVCV_BGR, NVCV_S16):
			//return NOS_FORMAT_B16G16R16_SINT;
		case SwitchPair(NVCV_RGBA, NVCV_S16):
			return NOS_FORMAT_R16G16B16A16_SINT;
		case SwitchPair(NVCV_BGRA, NVCV_S16):
			//return NOS_FORMAT_R16G16B16A16_SINT;

		//In here we adhere to UNORM because MOSTLY deep learning models 
		//for images uses pixel values in the range of 0..1
		case SwitchPair(NVCV_A, NVCV_F16):
			return NOS_FORMAT_R16_SFLOAT;
		case SwitchPair(NVCV_RGB, NVCV_F16):
			return NOS_FORMAT_R16G16B16_SFLOAT;
		case SwitchPair(NVCV_BGR, NVCV_F16):
			//return NOS_FORMAT_B16G16R16_SFLOAT;
		case SwitchPair(NVCV_RGBA, NVCV_F16):
			return NOS_FORMAT_R16G16B16A16_SFLOAT;
		case SwitchPair(NVCV_BGRA, NVCV_F16):
			//return NOS_FORMAT_B16G16R16A16_SFLOAT;
	

		case SwitchPair(NVCV_A, NVCV_U32):
			return NOS_FORMAT_R32_UINT;
		case SwitchPair(NVCV_RGB, NVCV_U32):
			//return NOS_FORMAT_R32G32B32_UNORM;
		case SwitchPair(NVCV_BGR, NVCV_U32):
			//return NOS_FORMAT_B32G32R32_UINT;
		case SwitchPair(NVCV_RGBA, NVCV_U32):
			return NOS_FORMAT_R32G32B32A32_UINT;
		case SwitchPair(NVCV_BGRA, NVCV_U32):
			//return NOS_FORMAT_B32G32R32A32_UINT;


		case SwitchPair(NVCV_A, NVCV_S32):
			return NOS_FORMAT_R32_SINT;
		case SwitchPair(NVCV_RGB, NVCV_S32):
			return NOS_FORMAT_R32G32B32_SINT;
		case SwitchPair(NVCV_BGR, NVCV_S32):
			//return NOS_FORMAT_B32G32R32_SINT;
		case SwitchPair(NVCV_RGBA, NVCV_S32):
			return NOS_FORMAT_R32G32B32A32_SINT;
		case SwitchPair(NVCV_BGRA, NVCV_S32):
			//return NOS_FORMAT_B32G32R32A32_SINT;
	
		case SwitchPair(NVCV_A, NVCV_F32):
			return NOS_FORMAT_R32_SFLOAT;
		case SwitchPair(NVCV_RGB, NVCV_F32):
			return NOS_FORMAT_R32G32B32_SFLOAT;
		case SwitchPair(NVCV_BGR, NVCV_F32):
			//return NOS_FORMAT_B32G32R32_SFLOAT;
		case SwitchPair(NVCV_RGBA, NVCV_F32):
			return NOS_FORMAT_R32G32B32A32_SFLOAT;
		case SwitchPair(NVCV_BGRA, NVCV_F32):
			//return NOS_FORMAT_B32G32R32A32_SFLOAT;

		case SwitchPair(NVCV_A, NVCV_U64):
			//return NOS_FORMAT_R64_UINT;
		case SwitchPair(NVCV_RGB, NVCV_U64):
			//return NOS_FORMAT_R64G64B64_UINT;
		case SwitchPair(NVCV_BGR, NVCV_U64):
			//return NOS_FORMAT_B64G64R64_UINT;
		case SwitchPair(NVCV_RGBA, NVCV_U64):
			//return NOS_FORMAT_R64G64B64A64_UINT;
		case SwitchPair(NVCV_BGRA, NVCV_U64):
			//return NOS_FORMAT_B64G64R64A64_UINT;


		case SwitchPair(NVCV_A, NVCV_S64):
			//return NOS_FORMAT_R64_SINT;
		case SwitchPair(NVCV_RGB, NVCV_S64):
			//return NOS_FORMAT_R64G64B64_SINT;
		case SwitchPair(NVCV_BGR, NVCV_S64):
			//return NOS_FORMAT_B64G64R64_SINT;
		case SwitchPair(NVCV_RGBA, NVCV_S64):
			//return NOS_FORMAT_R64G64B64A64_SINT;
		case SwitchPair(NVCV_BGRA, NVCV_S64):
			//return NOS_FORMAT_B64G64R64A64_SINT;


		case SwitchPair(NVCV_A, NVCV_F64):
			//return NOS_FORMAT_R64_SFLOAT;
		case SwitchPair(NVCV_RGB, NVCV_F64):
			//return NOS_FORMAT_R64G64B64_SFLOAT;
		case SwitchPair(NVCV_BGR, NVCV_F64):
			//return NOS_FORMAT_B64G64R64_SFLOAT;
		case SwitchPair(NVCV_RGBA, NVCV_F64):
			//return NOS_FORMAT_R64G64B64A64_SFLOAT;
		case SwitchPair(NVCV_BGRA, NVCV_F64):
			//return NOS_FORMAT_B64G64R64A64_SFLOAT;
	}
	return NOS_FORMAT_NONE;
}
