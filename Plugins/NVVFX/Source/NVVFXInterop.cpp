#include "NVVFXInterop.h"
#include "Nodos/Helpers.hpp"
#include "nosVulkanSubsystem/Helpers.hpp"
#include <Windows.h>


CUDAVulkanInterop::CUDAVulkanInterop()
{
	int deviceCount = 0;
	nosCUDA->GetDeviceCount(&deviceCount);
	if (deviceCount != 0) {
		nosCUDA->Initialize(0);
	}
}
CUDAVulkanInterop::~CUDAVulkanInterop()
{
}
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

void CUDAVulkanInterop::SetCUDAMemoryToVulkan(int64_t cudaPointerAddress, int width, int height ,size_t blockSize, size_t allocationSize, size_t offset, int32_t format, nosResourceShareInfo* outNosTexture) {
	outNosTexture->Info.Type = NOS_RESOURCE_TYPE_BUFFER;
	outNosTexture->Info.Buffer.Size = allocationSize;
	outNosTexture->Info.Buffer.Usage = nosBufferUsage(nosBufferUsage::NOS_BUFFER_USAGE_TRANSFER_SRC | nosBufferUsage::NOS_BUFFER_USAGE_TRANSFER_DST);

	outNosTexture->Memory.Handle = 0;
	outNosTexture->Memory.ExternalMemory.HandleType = NOS_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
	outNosTexture->Memory.ExternalMemory.Handle = cudaPointerAddress;
	outNosTexture->Memory.ExternalMemory.PID = getpid();
	outNosTexture->Memory.Offset = offset;
	outNosTexture->Memory.ExternalMemory.AllocationSize = blockSize;
	outNosTexture->Memory.Size = allocationSize;
}

nosResult CUDAVulkanInterop::AllocateNVCVImage(std::string name, int width, int height, NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType, size_t size, int planar, NvCVImage* out)
{
	nosCUDABufferInfo buf = {};
	nosResult res = nosCUDA->CreateShareableBufferOnCUDA(&buf, size);
	if (res != NOS_RESULT_SUCCESS) {
		return res;
	}

	nosFormat temp = {};
	temp = GetVulkanFormatFromNVCVImage(pixelFormat, compType);
	
	int componentNum = GetComponentNumFromVulkanFormat(temp);
	int componentByte = GetComponentBytesFromVulkanFormat(temp);

	out->width = width;
	out->height = height;
	out->pitch = (planar == NVCV_PLANAR) ? (width * componentByte) : (width*componentByte*componentNum);
	out->pixelFormat = pixelFormat;
	out->componentType = compType;
	out->pixelBytes = componentByte * componentNum;
	out->componentBytes = componentByte;
	out->numComponents = componentNum;
	out->pixels = reinterpret_cast<void*>(buf.Address);
	out->gpuMem = NVCV_CUDA;
	out->bufferBytes = buf.CreateInfo.AllocationSize;
	out->planar = planar;

	return NOS_RESULT_SUCCESS;
}

nosResult CUDAVulkanInterop::AllocateShareableNVCVImage(std::string name, int width, int height, NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType, size_t size, int planar, NvCVImage* out)
{
	nosCUDABufferInfo buf = {};
	nosResult res = nosCUDA->CreateShareableBufferOnCUDA(&buf, size);
	if (res != NOS_RESULT_SUCCESS) {
		return res;
	}

	nosFormat temp = {};
	temp = GetVulkanFormatFromNVCVImage(pixelFormat, compType);

	int componentNum = GetComponentNumFromVulkanFormat(temp);
	int componentByte = GetComponentBytesFromVulkanFormat(temp);

	out->width = width;
	out->height = height;
	out->pitch = (planar == NVCV_PLANAR) ? (width * componentByte) : (width * componentByte * componentNum);
	out->pixelFormat = pixelFormat;
	out->componentType = compType;
	out->pixelBytes = componentByte * componentNum;
	out->componentBytes = componentByte;
	out->numComponents = componentNum;
	out->pixels = reinterpret_cast<void*>(buf.Address);
	out->gpuMem = NVCV_CUDA;
	out->bufferBytes = buf.CreateInfo.AllocationSize;
	out->planar = planar;

	return NOS_RESULT_SUCCESS;
}

nosResult CUDAVulkanInterop::nosTextureToNVCVImage(nosResourceShareInfo& vulkanTex, NvCVImage& nvcvImage, std::optional<nosNVCVLayout> layout)
{
	int componentNum = GetComponentNumFromVulkanFormat(vulkanTex.Info.Texture.Format);
	int componentByte = GetComponentBytesFromVulkanFormat(vulkanTex.Info.Texture.Format);

	if ((vulkanTex.Info.Texture.Width * vulkanTex.Info.Texture.Height * componentByte * componentNum) == vulkanTexBuf.Info.Buffer.Size) {
		if (vulkanTexBuf.Memory.ExternalMemory.Handle != NULL) {
			nosCmd texToBuf = {};
			nosGPUEvent waitTexToBuf = {};
			nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitTexToBuf };
			nosVulkan->Begin("TexToBuf", &texToBuf);
			nosVulkan->Copy(texToBuf, &vulkanTex, &vulkanTexBuf, 0);
			nosVulkan->End(texToBuf, &endParams);
			nosVulkan->WaitGpuEvent(&waitTexToBuf, UINT64_MAX);
			return NOS_RESULT_SUCCESS;
		}
	}

	if (vulkanTexBuf.Memory.Handle != NULL) {
		nosVulkan->DestroyResource(&vulkanTexBuf);
	}

	uint64_t gpuPointer = 0;

	vulkanTexBuf.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
	vulkanTexBuf.Info.Buffer.Size = vulkanTex.Info.Texture.Width * vulkanTex.Info.Texture.Height * componentByte * componentNum;
	vulkanTexBuf.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
	nosVulkan->CreateResource(&vulkanTexBuf);

	nosCmd texToBuf = {};
	nosGPUEvent waitTexToBuf = {};
	nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitTexToBuf };
	nosVulkan->Begin("TexToBuf", &texToBuf);
	nosVulkan->Copy(texToBuf, &vulkanTex, &vulkanTexBuf, 0);
	nosVulkan->End(texToBuf, &endParams);
	nosVulkan->WaitGpuEvent(&waitTexToBuf, UINT64_MAX);

	void* vulkanData = nosVulkan->Map(&vulkanTexBuf);
	nosCUDABufferInfo cudaBuf = {};
	nosResult res = nosCUDA->ImportExternalMemoryAsCUDABuffer(vulkanTexBuf.Memory.ExternalMemory.Handle, vulkanTexBuf.Memory.ExternalMemory.AllocationSize, 
		vulkanTexBuf.Memory.Size, vulkanTexBuf.Memory.Offset, EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUEWIN32, &cudaBuf);


	gpuPointer = cudaBuf.Address;

	if (res != NOS_RESULT_SUCCESS)
		return res;

	nvcvImage.width = vulkanTex.Info.Texture.Width;
	nvcvImage.height = vulkanTex.Info.Texture.Height;
	nvcvImage.pitch = vulkanTex.Info.Texture.Width * componentByte * componentNum; //There is no planar vulkan texture at the moment?
	nvcvImage.pixelFormat = GetPixelFormatFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.componentType = GetComponentTypeFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.pixelBytes = componentByte * componentNum;
	nvcvImage.componentBytes = componentByte;
	nvcvImage.numComponents = componentNum;
	nvcvImage.pixels = reinterpret_cast<void*>(gpuPointer);
	nvcvImage.gpuMem = NVCV_CUDA;
	nvcvImage.bufferBytes = vulkanTexBuf.Memory.Size;
	nvcvImage.planar = (layout == std::nullopt) ? (nvcvImage.planar) : ((int)layout.value());
	
	return res;
}

nosResult CUDAVulkanInterop::NVCVImageToNosTexture(NvCVImage& nvcvImage, nosResourceShareInfo& vulkanTex, std::optional<nosNVCVLayout> layout)
{
	int64_t size = nvcvImage.bufferBytes;
	if (size <= 0) {
		nosEngine.LogE("NVCV Image size %d can not be converted to nosTexture!", size);
		return NOS_RESULT_FAILED;
	}
	uint64_t shareableHandle = NULL;
	nosCUDABufferInfo buff = {};
	nosResult res = nosCUDA->GetCUDABufferFromAddress(reinterpret_cast<uint64_t>(nvcvImage.pixels), &buff);
	//nosResult res = GPUResManager.GetShareableHandle(reinterpret_cast<uint64_t>(nvcvImage.pixels), &shareableHandle);
	if (res != NOS_RESULT_SUCCESS)
		return res;
	shareableHandle = buff.ShareInfo.ShareableHandle;
	SetCUDAMemoryToVulkan(shareableHandle, nvcvImage.width, nvcvImage.height,
		size, size, 0, GetVulkanFormatFromNVCVImage(nvcvImage), &vulkanTex);
	
	res = nosVulkan->ImportResource(&vulkanTex);

	return res;
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

int CUDAVulkanInterop::GetComponentBytesFromVulkanFormat(nosFormat format)
{
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
		return 1;

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
		return 2;

	case NOS_FORMAT_R32G32_UINT:
	case NOS_FORMAT_R32G32B32_UINT:
	case NOS_FORMAT_R32G32B32A32_UINT:
	case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
	case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
	case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
	case NOS_FORMAT_X8_D24_UNORM_PACK32:
		return 4;

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
		return 4;

	case NOS_FORMAT_R16_SFLOAT:
	case NOS_FORMAT_R16G16_SFLOAT:
	case NOS_FORMAT_R16G16B16_SFLOAT:
	case NOS_FORMAT_R16G16B16A16_SFLOAT:
		return 2;

	case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
	case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
	case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
	case NOS_FORMAT_R32_SINT:
	case NOS_FORMAT_R32G32_SINT:
	case NOS_FORMAT_R32G32B32_SINT:
	case NOS_FORMAT_R32G32B32A32_SINT:
		return 4;

	case NOS_FORMAT_R32_SFLOAT:
	case NOS_FORMAT_R32G32_SFLOAT:
	case NOS_FORMAT_R32G32B32_SFLOAT:
	case NOS_FORMAT_R32G32B32A32_SFLOAT:
	case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
	case NOS_FORMAT_D32_SFLOAT:
		return 4;

		//Cant be mapped for now
		//return NvCVImage_ComponentType::NVCV_U64;
		//return NvCVImage_ComponentType::NVCV_S64;
		//return NvCVImage_ComponentType::NVCV_F64;

	}
	return 0;
}
int CUDAVulkanInterop::GetComponentNumFromVulkanFormat(nosFormat format)
{
	switch (format) {
	case NOS_FORMAT_NONE:
	case NOS_FORMAT_R8_UNORM:
	case NOS_FORMAT_R8_UINT:
	case NOS_FORMAT_R8_SRGB:
	case NOS_FORMAT_R16_UNORM:
	case NOS_FORMAT_R16_SNORM:
	case NOS_FORMAT_R16_USCALED:
	case NOS_FORMAT_R16_SSCALED:
	case NOS_FORMAT_R16_UINT:
	case NOS_FORMAT_R16_SINT:
	case NOS_FORMAT_R16_SFLOAT:
	case NOS_FORMAT_R32_UINT:
	case NOS_FORMAT_R32_SINT:
	case NOS_FORMAT_R32_SFLOAT:
	case NOS_FORMAT_D16_UNORM:
	case NOS_FORMAT_D32_SFLOAT:
		return 1;
	case NOS_FORMAT_R8G8_UNORM:
	case NOS_FORMAT_R8G8_UINT:
	case NOS_FORMAT_R8G8_SRGB:
	case NOS_FORMAT_R16G16_UNORM:
	case NOS_FORMAT_R16G16_SNORM:
	case NOS_FORMAT_R16G16_USCALED:
	case NOS_FORMAT_R16G16_SSCALED:
	case NOS_FORMAT_R16G16_UINT:
	case NOS_FORMAT_R16G16_SINT:
	case NOS_FORMAT_R16G16_SFLOAT:
	case NOS_FORMAT_R32G32_UINT:
	case NOS_FORMAT_R32G32_SINT:
	case NOS_FORMAT_R32G32_SFLOAT:
		return 2;
	case NOS_FORMAT_R8G8B8_UNORM:
	case NOS_FORMAT_R8G8B8_SRGB:
	case NOS_FORMAT_B8G8R8_UNORM:
	case NOS_FORMAT_B8G8R8_UINT:
	case NOS_FORMAT_B8G8R8_SRGB:
	case NOS_FORMAT_R16G16B16_UNORM:
	case NOS_FORMAT_R16G16B16_SNORM:
	case NOS_FORMAT_R16G16B16_USCALED:
	case NOS_FORMAT_R16G16B16_SSCALED:
	case NOS_FORMAT_R16G16B16_UINT:
	case NOS_FORMAT_R16G16B16_SINT:
	case NOS_FORMAT_R16G16B16_SFLOAT:
	case NOS_FORMAT_R32G32B32_UINT:
	case NOS_FORMAT_R32G32B32_SINT:
	case NOS_FORMAT_R32G32B32_SFLOAT:
	case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
	case NOS_FORMAT_G8B8G8R8_422_UNORM:
	case NOS_FORMAT_B8G8R8G8_422_UNORM:
		return 3;
	case NOS_FORMAT_R8G8B8A8_UNORM:
	case NOS_FORMAT_R8G8B8A8_UINT:
	case NOS_FORMAT_R8G8B8A8_SRGB:
	case NOS_FORMAT_B8G8R8A8_UNORM:
	case NOS_FORMAT_B8G8R8A8_SRGB:
	case NOS_FORMAT_R16G16B16A16_UNORM:
	case NOS_FORMAT_R16G16B16A16_SNORM:
	case NOS_FORMAT_R16G16B16A16_USCALED:
	case NOS_FORMAT_R16G16B16A16_SSCALED:
	case NOS_FORMAT_R16G16B16A16_UINT:
	case NOS_FORMAT_R16G16B16A16_SINT:
	case NOS_FORMAT_R16G16B16A16_SFLOAT:
	case NOS_FORMAT_R32G32B32A32_UINT:
	case NOS_FORMAT_R32G32B32A32_SINT:
	case NOS_FORMAT_R32G32B32A32_SFLOAT:
	case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
	case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
	case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
	case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
	case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
	case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
		return 4;
	default:
		return 0;
	}
}


//A simple trick to handle multi variable switch statement
constexpr uint64_t SwitchPair(NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType) {
	return ((pixelFormat << 16) + compType);
}

nosFormat CUDAVulkanInterop::GetVulkanFormatFromNVCVImage(NvCVImage nvcvImage)
{
	return GetVulkanFormatFromNVCVImage(nvcvImage.pixelFormat, nvcvImage.componentType);
}

nosFormat CUDAVulkanInterop::GetVulkanFormatFromNVCVImage(NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType componentType)
{
	switch (SwitchPair(pixelFormat, componentType)) {
	case SwitchPair(NVCV_A, NVCV_U8):
		return NOS_FORMAT_R8_UINT;
	case SwitchPair(NVCV_RGB, NVCV_U8):
		//return NOS_FORMAT_R8G8B8_UINT;
	case SwitchPair(NVCV_BGR, NVCV_U8):
		return NOS_FORMAT_B8G8R8_UINT;
	case SwitchPair(NVCV_RGBA, NVCV_U8):
		return NOS_FORMAT_R8G8B8A8_UINT;
	case SwitchPair(NVCV_BGRA, NVCV_U8):
		return NOS_FORMAT_B8G8R8A8_UNORM;

	case SwitchPair(NVCV_A, NVCV_U16):
		return NOS_FORMAT_R16_UINT;
	case SwitchPair(NVCV_RGB, NVCV_U16):
	case SwitchPair(NVCV_BGR, NVCV_U16):
		return NOS_FORMAT_R16G16B16_UINT;
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
	case SwitchPair(NVCV_BGR, NVCV_F16):
		return NOS_FORMAT_R16G16B16_SFLOAT;
		//return NOS_FORMAT_B16G16R16_SFLOAT;
	case SwitchPair(NVCV_RGBA, NVCV_F16):
		return NOS_FORMAT_R16G16B16A16_SFLOAT;
	case SwitchPair(NVCV_BGRA, NVCV_F16):
		//return NOS_FORMAT_B16G16R16A16_SFLOAT;


	case SwitchPair(NVCV_A, NVCV_U32):
		return NOS_FORMAT_R32_UINT;
	case SwitchPair(NVCV_RGB, NVCV_U32):
	case SwitchPair(NVCV_BGR, NVCV_U32):
		return NOS_FORMAT_R32G32B32_UINT;
		//return NOS_FORMAT_B32G32R32_UINT;
	case SwitchPair(NVCV_RGBA, NVCV_U32):
		return NOS_FORMAT_R32G32B32A32_UINT;
	case SwitchPair(NVCV_BGRA, NVCV_U32):
		//return NOS_FORMAT_B32G32R32A32_UINT;


	case SwitchPair(NVCV_A, NVCV_S32):
		return NOS_FORMAT_R32_SINT;
	case SwitchPair(NVCV_RGB, NVCV_S32):
	case SwitchPair(NVCV_BGR, NVCV_S32):
		return NOS_FORMAT_R32G32B32_SINT;
		//return NOS_FORMAT_B32G32R32_SINT;
	case SwitchPair(NVCV_RGBA, NVCV_S32):
		return NOS_FORMAT_R32G32B32A32_SINT;
	case SwitchPair(NVCV_BGRA, NVCV_S32):
		//return NOS_FORMAT_B32G32R32A32_SINT;

	case SwitchPair(NVCV_A, NVCV_F32):
		return NOS_FORMAT_R32_SFLOAT;
	case SwitchPair(NVCV_RGB, NVCV_F32):
	case SwitchPair(NVCV_BGR, NVCV_F32):
		return NOS_FORMAT_R32G32B32_SFLOAT;
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
		break;
	}
	return NOS_FORMAT_NONE;
}

void CUDAVulkanInterop::NormalizeNVCVImage(NvCVImage* nvcvImage)
{
	dim3 threadsPerBlock(256);
	dim3 numBlocks((nvcvImage->width * nvcvImage->height + threadsPerBlock.x - 1) / threadsPerBlock.x);

//	float* before = new float[nvcvImage->bufferBytes];
//	cudaError res = cudaMemcpy(before, nvcvImage->pixels, nvcvImage->bufferBytes, cudaMemcpyDeviceToHost);
//	assert(res == cudaSuccess);
	//NormalizeKernelWrapper(numBlocks, threadsPerBlock, reinterpret_cast<float*>(nvcvImage->pixels), 255, nvcvImage->bufferBytes);
	
//	float* after = new float[nvcvImage->bufferBytes];
//	res = cudaMemcpy(after, nvcvImage->pixels, nvcvImage->bufferBytes, cudaMemcpyDeviceToHost);
//	assert(res == cudaSuccess);
	
	//delete[] before;
	//delete[] after;

}

void CUDAVulkanInterop::CopyNVCVImages(NvCVImage* src, NvCVImage* dst)
{
	nosCUDABufferInfo srcBuf = {}, dstBuf = {};
	nosCUDA->GetCUDABufferFromAddress(reinterpret_cast<uint64_t>(src->pixels), &srcBuf);
	nosCUDA->GetCUDABufferFromAddress(reinterpret_cast<uint64_t>(dst->pixels), &dstBuf);
	nosCUDA->CopyBuffers(&srcBuf, &dstBuf);
}

//TODO: USE THESE!!!
/*
cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
{
cudaChannelFormatDesc d;
memset(&d, 0, sizeof(d));
switch (format) {
case VK_FORMAT_R8_UINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R8_SINT: d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
case VK_FORMAT_R8G8_UINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R8G8_SINT: d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
case VK_FORMAT_R8G8B8A8_UINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R8G8B8A8_SINT: d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindSigned; break;
case VK_FORMAT_R16_UINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R16_SINT: d.x = 16; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindSigned; break;
case VK_FORMAT_R16G16_UINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R16G16_SINT: d.x = 16; d.y = 16; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindSigned; break;
(continues on next page)
106 Chapter 6. Programming Interface
CUDA C++ Programming Guide, Release 12.3
(continued from previous page)
case VK_FORMAT_R16G16B16A16_UINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f =
cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R16G16B16A16_SINT: d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f =
cudaChannelFormatKindSigned; break;
case VK_FORMAT_R32_UINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R32_SINT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindSigned; break;
case VK_FORMAT_R32_SFLOAT: d.x = 32; d.y = 0; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindFloat; break;
case VK_FORMAT_R32G32_UINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R32G32_SINT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindSigned; break;
case VK_FORMAT_R32G32_SFLOAT: d.x = 32; d.y = 32; d.z = 0; d.w = 0; d.f =
cudaChannelFormatKindFloat; break;
case VK_FORMAT_R32G32B32A32_UINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f =
cudaChannelFormatKindUnsigned; break;
case VK_FORMAT_R32G32B32A32_SINT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f =
cudaChannelFormatKindSigned; break;
case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f =
cudaChannelFormatKindFloat; break;
default: assert(0);
}
return d;
}
cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers,
,→VkImageViewType vkImageViewType) {
cudaExtent e = { 0, 0, 0 };
switch (vkImageViewType) {
case VK_IMAGE_VIEW_TYPE_1D: e.width = vkExt.width; e.height = 0;
,→ e.depth = 0; break;
case VK_IMAGE_VIEW_TYPE_2D: e.width = vkExt.width; e.height = vkExt.
,→height; e.depth = 0; break;
case VK_IMAGE_VIEW_TYPE_3D: e.width = vkExt.width; e.height = vkExt.
,→height; e.depth = vkExt.depth; break;
case VK_IMAGE_VIEW_TYPE_CUBE: e.width = vkExt.width; e.height = vkExt.
,→height; e.depth = arrayLayers; break;
case VK_IMAGE_VIEW_TYPE_1D_ARRAY: e.width = vkExt.width; e.height = 0;
,→ e.depth = arrayLayers; break;
case VK_IMAGE_VIEW_TYPE_2D_ARRAY: e.width = vkExt.width; e.height = vkExt.
,→height; e.depth = arrayLayers; break;
case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.
,→height; e.depth = arrayLayers; break;
default: assert(0);
}
return e;
}
unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType,
,→ VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore) {
unsigned int flags = 0;
(continues on next page)
6.2. CUDA Runtime 107
CUDA C++ Programming Guide, Release 12.3
(continued from previous page)
switch (vkImageViewType) {
case VK_IMAGE_VIEW_TYPE_CUBE: flags |= cudaArrayCubemap;
,→break;
case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered;
,→break;
case VK_IMAGE_VIEW_TYPE_1D_ARRAY: flags |= cudaArrayLayered;
,→break;
case VK_IMAGE_VIEW_TYPE_2D_ARRAY: flags |= cudaArrayLayered;
,→break;
default: break;
}
if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
flags |= cudaArrayColorAttachment;
}
if (allowSurfaceLoadStore) {
flags |= cudaArraySurfaceLoadStore;
}
return flags;
}


*/