#include "CUDAVulkanInterop.h"
#include "Nodos/Helpers.hpp"
#include "nosVulkanSubsystem/Helpers.hpp"
#include <Windows.h>
#include "CommonCUDAKernels.h"

CUDAVulkanInterop::CUDAVulkanInterop()
{
	InitCUDA();
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
nosResult CUDAVulkanInterop::SetVulkanMemoryToCUDA(int64_t handle, size_t blockSize, size_t allocationSize, size_t offset, uint64_t* outCudaPointerAddres)
{

	cudaExternalMemory_t externalMemory;
	cudaExternalMemoryHandleDesc desc;
	memset(&desc, 0, sizeof(desc));

	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	desc.handle.win32.handle = reinterpret_cast<void*>(handle);
	desc.handle.win32.name = NULL;
	desc.size = blockSize;
	
	cudaError res = cudaImportExternalMemory(&externalMemory, &desc);
	
	if (res != CUDA_SUCCESS) {
		nosEngine.LogE("Import attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}

	void* pointer = nullptr;

	cudaExternalMemoryBufferDesc bufferDesc;
	bufferDesc.flags = 0; // must be zero
	bufferDesc.offset = offset; //not working for non zero offsets
	bufferDesc.size = allocationSize;

	res = cudaExternalMemoryGetMappedBuffer(&pointer, externalMemory, &bufferDesc);
	cudaError syncRes = cudaDeviceSynchronize();
	if (syncRes != CUDA_SUCCESS) {
		nosEngine.LogE("CUDA device synchronize failed with error code %d", syncRes);
		return NOS_RESULT_FAILED;
	}

	if (res != CUDA_SUCCESS || pointer == NULL) {
		nosEngine.LogE("Get mapped buffer attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}
	
	(*outCudaPointerAddres) = reinterpret_cast<uint64_t>(pointer);
	return NOS_RESULT_SUCCESS;
}

void CUDAVulkanInterop::SetCUDAMemoryToVulkan(int64_t cudaPointerAddress, int width, int height ,size_t blockSize, size_t allocationSize, size_t offset, int32_t format, nos::sys::vulkan::TTexture* outNosTexture) {
	outNosTexture->size = nos::sys::vulkan::SizePreset::CUSTOM;
	outNosTexture->width = width;
	outNosTexture->height = height;
	outNosTexture->format = nos::sys::vulkan::Format(format);
				 
	outNosTexture->usage |= nos::sys::vulkan::ImageUsage::SAMPLED | nos::sys::vulkan::ImageUsage::TRANSFER_DST | nos::sys::vulkan::ImageUsage::TRANSFER_SRC;
	outNosTexture->unmanaged = false;
	outNosTexture->unscaled = true;
	outNosTexture->handle = 0;
				 
	outNosTexture->external_memory.mutate_handle_type(0x00000002);
	outNosTexture->external_memory.mutate_handle(cudaPointerAddress);
	outNosTexture->external_memory.mutate_pid(getpid());
	outNosTexture->external_memory.mutate_offset(offset);
	outNosTexture->external_memory.mutate_allocation_size(allocationSize);
	outNosTexture->external_memory.mutate_block_size(blockSize);
}

nosResult CUDAVulkanInterop::AllocateNVCVImage(std::string name, int width, int height, NvCVImage_PixelFormat pixelFormat, NvCVImage_ComponentType compType, size_t size, NvCVImage* out)
{
	int64_t handle = 0;
	handle = GPUResManager.AllocateShareableGPU(name, size);

	if (handle == NULL) {
		nosEngine.LogE("cuMemExportToShareableHandle failed!");
		return NOS_RESULT_FAILED;
	}
	uint64_t buffer = NULL;
	nosResult res = GPUResManager.GetGPUBuffer(name, &buffer);
 	uint64_t realSize = GPUResManager.GetSize(name);

	/*float a = 1.0f;

	int b = std::bit_cast<int, float>(a);

	cudaSetDevice(0);
		= cudaMemset(reinterpret_cast<void*>(buffer), b, realSize);*/
	cudaError cudaRes = cudaSuccess;
	
	cudaError syncRes = cudaDeviceSynchronize();
	if (syncRes != CUDA_SUCCESS) {
		nosEngine.LogE("CUDA device synchronize failed with error code %d", syncRes);
		return NOS_RESULT_FAILED;
	}

	assert(cudaRes == cudaSuccess);
	
	if (res != NOS_RESULT_SUCCESS) {
		return res;
	}

	nosFormat temp = {};
	temp = GetVulkanFormatFromNVCVImage(pixelFormat, compType);
	
	int componentNum = GetComponentNumFromVulkanFormat(temp);
	int componentByte = GetComponentBytesFromVulkanFormat(temp);

	out->width = width;
	out->height = height;
	out->pitch = width * componentByte;
	out->pixelFormat = pixelFormat;
	out->componentType = compType;
	out->pixelBytes = componentByte * componentNum;
	out->componentBytes = componentByte;
	out->numComponents = componentNum;
	out->pixels = reinterpret_cast<void*>(buffer);
	out->gpuMem = NVCV_CUDA;
	out->bufferBytes = realSize;
	out->planar = NVCV_INTERLEAVED;

	return NOS_RESULT_SUCCESS;
}

nosResult CUDAVulkanInterop::nosTextureToNVCVImage(nosResourceShareInfo& vulkanTex, NvCVImage& nvcvImage, std::optional<nosNVCVLayout> layout)
{
	uint64_t gpuPointer = 0;
	size_t textureSize2 = vulkanTex.Info.Texture.GetSize();
	size_t textureSize;
	nosVulkan->GetImageSize(vulkanTex.Memory.Handle, &textureSize);
	nosResult res = SetVulkanMemoryToCUDA(vulkanTex.Memory.ExternalMemory.Handle, vulkanTex.Memory.ExternalMemory.BlockSize, vulkanTex.Memory.ExternalMemory.AllocationSize, vulkanTex.Memory.ExternalMemory.Offset, &gpuPointer);


	if (res != NOS_RESULT_SUCCESS)
		return res;

	int componentNum = GetComponentNumFromVulkanFormat(vulkanTex.Info.Texture.Format);
	int componentByte = GetComponentBytesFromVulkanFormat(vulkanTex.Info.Texture.Format);

	nvcvImage.width = vulkanTex.Info.Texture.Width;
	nvcvImage.height = vulkanTex.Info.Texture.Height;
	nvcvImage.pitch = vulkanTex.Info.Texture.Width * componentByte;
	nvcvImage.pixelFormat = GetPixelFormatFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.componentType = GetComponentTypeFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.pixelBytes = componentByte * componentNum;
	nvcvImage.componentBytes = componentByte;
	nvcvImage.numComponents = componentNum;
	nvcvImage.pixels = reinterpret_cast<void*>(gpuPointer);
	nvcvImage.gpuMem = NVCV_CUDA;
	nvcvImage.bufferBytes = vulkanTex.Memory.ExternalMemory.AllocationSize;
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
	nosResult res = GPUResManager.GetShareableHandle(reinterpret_cast<uint64_t>(nvcvImage.pixels), &shareableHandle);

	/*void* hostBufferNormal = malloc(std::min(size, size2));
	

	//cudaMemset(nvcvImage.pixels, 255, std::min(size, size2))
	
	//GPUResManager.MemCopy(reinterpret_cast<int64_t>(nvcvImage.pixels), "Trial", std::min(size,size2));

	/*status = cuMemExportToShareableHandle(&shareableHandle, handle, CU_MEM_HANDLE_TYPE_WIN32, 0);*/

	nos::sys::vulkan::TTexture tex;

	SetCUDAMemoryToVulkan(shareableHandle, nvcvImage.width, nvcvImage.height,
		size, size, 0, GetVulkanFormatFromNVCVImage(nvcvImage), &tex);
	
	vulkanTex = nos::vkss::ConvertDef(tex);
	res = nosVulkan->ImportResource(&vulkanTex);

	return res;
}

void CUDAVulkanInterop::InitCUDA()
{
	CUcontext ctx;
	CUdevice dev;
	int supportsVMM = 0;
	CUresult status = cuInit(0);
	if (status != CUDA_SUCCESS) {
		nosEngine.LogE("cuInit failed!");
	}

	// Get the first CUDA device and create a context

	status = cuDevicePrimaryCtxRetain(&ctx, 0);
	if (status != CUDA_SUCCESS) {
		nosEngine.LogE("cuDevicePrimaryCtxRetain failed!");
	}

	status = cuCtxSetCurrent(ctx);
	if (status != CUDA_SUCCESS) {
		nosEngine.LogE("cuCtxSetCurrent failed!");
	}

	cuCtxGetDevice(&dev);
	if (status != CUDA_SUCCESS) {
		nosEngine.LogE("cuDeviceGet failed!");
	}
	// Check if the device supports Virtual Memory Management
	status = cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev);
	if (status != CUDA_SUCCESS || !supportsVMM) {
		nosEngine.LogE("Device does not support Virtual Memory Management!");
	}
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
		//return NOS_FORMAT_B8G8R8A8_UINT;

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
	NormalizeKernelWrapper(numBlocks, threadsPerBlock, reinterpret_cast<float*>(nvcvImage->pixels), 255, nvcvImage->bufferBytes);
	
//	float* after = new float[nvcvImage->bufferBytes];
//	res = cudaMemcpy(after, nvcvImage->pixels, nvcvImage->bufferBytes, cudaMemcpyDeviceToHost);
//	assert(res == cudaSuccess);
	
	//delete[] before;
	//delete[] after;

}

nosResult CUDAVulkanInterop::CopyNVCVImage(NvCVImage* src, NvCVImage* dst)
{
	return GPUResManager.MemCopy(reinterpret_cast<uint64_t>(src->pixels), reinterpret_cast<uint64_t>(dst->pixels), std::min(src->bufferBytes, dst->bufferBytes));
}
