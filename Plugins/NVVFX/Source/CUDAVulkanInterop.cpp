#include "CUDAVulkanInterop.h"
#include "Nodos/Helpers.hpp"
#include "nosVulkanSubsystem/Helpers.hpp"
#include <Windows.h>

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
nosResult CUDAVulkanInterop::SetVulkanMemoryToCUDA(int64_t handle, size_t size, size_t offset, int64_t& outCudaPointerAddres)
{

	outCudaPointerAddres = NULL;
	cudaExternalMemory_t externalMemory;
	cudaExternalMemoryHandleDesc desc;
	memset(&desc, 0, sizeof(desc));

	desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	desc.handle.win32.handle = reinterpret_cast<void*>(handle);
	desc.handle.win32.name = NULL;
	desc.size = size;
	
	cudaError res = cudaImportExternalMemory(&externalMemory, &desc);

	if (res != CUDA_SUCCESS) {
		nosEngine.LogE("Import attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}

	void* pointer = nullptr;

	cudaExternalMemoryBufferDesc bufferDesc;
	bufferDesc.flags = 0; // must be zero
	bufferDesc.offset = offset;
	bufferDesc.size = size;

	res = cudaExternalMemoryGetMappedBuffer(&pointer, externalMemory, &bufferDesc);
	if (res != CUDA_SUCCESS || pointer == NULL) {
		nosEngine.LogE("Get mapped buffer attempt from Vulkan to CUDA failed for handle %d with Error Code %d", handle, res);
		return NOS_RESULT_FAILED;
	}
	
	outCudaPointerAddres = reinterpret_cast<uint64_t>(pointer);
	
	return NOS_RESULT_SUCCESS;
}

void CUDAVulkanInterop::SetCUDAMemoryToVulkan(int64_t cudaPointerAddress, int width, int height ,size_t size, size_t offset, int32_t format, nos::sys::vulkan::TTexture& outNosTexture) {
	outNosTexture.size = nos::sys::vulkan::SizePreset::CUSTOM;
	outNosTexture.width = width;
	outNosTexture.height = height;
	outNosTexture.format = nos::sys::vulkan::Format::R8G8B8A8_UNORM;

	outNosTexture.usage |= nos::sys::vulkan::ImageUsage::SAMPLED | nos::sys::vulkan::ImageUsage::TRANSFER_DST | nos::sys::vulkan::ImageUsage::TRANSFER_SRC;
	outNosTexture.unmanaged = false;
	outNosTexture.unscaled = true;
	outNosTexture.handle = 0;

	outNosTexture.external_memory.mutate_handle_type(0x00000002);
	outNosTexture.external_memory.mutate_handle(cudaPointerAddress);
	outNosTexture.external_memory.mutate_pid(getpid());
	outNosTexture.external_memory.mutate_offset(offset);
	outNosTexture.external_memory.mutate_allocation_size(size);
}

nosResult CUDAVulkanInterop::nosTextureToNVCVImage(nosResourceShareInfo& vulkanTex, NvCVImage& nvcvImage, std::optional<nosNVCVLayout> layout)
{
	int64_t gpuPointer = NULL;
	size_t textureSize2 = vulkanTex.Info.Texture.GetSize();
	size_t textureSize;
	nosVulkan->GetImageSize(vulkanTex.Memory.Handle, &textureSize);
	nosResult res = SetVulkanMemoryToCUDA(vulkanTex.Memory.ExternalMemory.Handle, vulkanTex.Memory.ExternalMemory.AllocationSize, vulkanTex.Memory.ExternalMemory.Offset, gpuPointer);


	if (res != NOS_RESULT_SUCCESS)
		return res;

	nvcvImage.pixels = reinterpret_cast<void*>(gpuPointer);
	nvcvImage.width = vulkanTex.Info.Texture.Width;
	nvcvImage.height = vulkanTex.Info.Texture.Height;
	nvcvImage.gpuMem = NVCV_CUDA;
	nvcvImage.pitch = textureSize2 / vulkanTex.Info.Texture.Height;
	nvcvImage.bufferBytes = textureSize;
	nvcvImage.pixelFormat = GetPixelFormatFromVulkanFormat(vulkanTex.Info.Texture.Format);
	nvcvImage.componentType = GetComponentTypeFromVulkanFormat(vulkanTex.Info.Texture.Format);
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

	cudaMemset(nvcvImage.pixels, 255, nvcvImage.bufferBytes);

	int64_t handle = 0;
	handle = GPUResManager.AllocateShareableGPU("Trial", size);

	int64_t size2 = GPUResManager.GetSize("Trial");
	cudaError resCu = cudaMemcpy(reinterpret_cast<void*>(GPUResManager.GetGPUBuffer("Trial")), nvcvImage.pixels, std::min(size, size2), cudaMemcpyDeviceToDevice);

	void* hostBufferNormal = malloc(std::min(size, size2));
	FILE* file2 = fopen("C:/TRASH/revealCudaNormal.txt", "wb");
	resCu = cudaMemcpy(hostBufferNormal, nvcvImage.pixels, std::min(size, size2), cudaMemcpyDeviceToHost);
	fwrite(hostBufferNormal, 1, std::min(size, size2), file2);
	fclose(file2);

	void* hostBuffer = malloc(std::min(size, size2));
	resCu = cudaMemcpy(hostBuffer, reinterpret_cast<void*>(GPUResManager.GetGPUBuffer("Trial")), std::min(size, size2), cudaMemcpyDeviceToHost);
	FILE* file = fopen("C:/TRASH/revealCudaShareable.txt", "wb");
	fwrite(hostBuffer, 1, std::min(size, size2), file);
	fclose(file);

	//cudaMemset(nvcvImage.pixels, 255, std::min(size, size2))
	
	//GPUResManager.MemCopy(reinterpret_cast<int64_t>(nvcvImage.pixels), "Trial", std::min(size,size2));

	/*status = cuMemExportToShareableHandle(&shareableHandle, handle, CU_MEM_HANDLE_TYPE_WIN32, 0);*/
	
	if (handle == NULL) {
		nosEngine.LogE("cuMemExportToShareableHandle failed!");
		return NOS_RESULT_FAILED;
	}
	
	nos::sys::vulkan::TTexture tex;

	SetCUDAMemoryToVulkan(handle, nvcvImage.width, nvcvImage.height,
		size2, vulkanTex.Memory.ExternalMemory.Offset, GetVulkanFormatFromNVCVImage(nvcvImage), tex);
	
	vulkanTex = nos::vkss::ConvertDef(tex);
	nosResult res = nosVulkan->ImportResource(&vulkanTex);

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
			break;
	}
	return NOS_FORMAT_NONE;
}
