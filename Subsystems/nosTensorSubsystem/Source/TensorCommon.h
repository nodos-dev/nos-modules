#pragma once
#ifndef TENSOR_SUBSYS_COMMON_H_INCLUDED
#define TENSOR_SUBSYS_COMMON_H_INCLUDED
#include "nosTensorSubsystem/nosTensorSubsystem.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"

namespace nos::tensor {
	static short GetSizeOfElementType(TensorElementType type) {
		switch (type) {
			case ELEMENT_TYPE_UNDEFINED:
				return 0;
				break;
			case ELEMENT_TYPE_FLOAT:
				return sizeof(float);
				break;
			case ELEMENT_TYPE_UINT8:
				return sizeof(uint8_t);
				break;
			case ELEMENT_TYPE_INT8:
				return sizeof(int8_t);
				break;
			case ELEMENT_TYPE_UINT16:
				return sizeof(uint16_t);
				break;
			case ELEMENT_TYPE_INT16:
				return sizeof(int16_t);
				break;
			case ELEMENT_TYPE_INT32:
				return sizeof(int32_t);
				break;
			case ELEMENT_TYPE_INT64:
				return sizeof(int64_t);
				break;
			case ELEMENT_TYPE_STRING:
				return sizeof(char); // ???? not sure
				break;
			case ELEMENT_TYPE_BOOL:
				return sizeof(bool);
				break;
			case ELEMENT_TYPE_FLOAT16:
				return sizeof(uint16_t);
				break;
			case ELEMENT_TYPE_DOUBLE:
				return sizeof(double);
				break;
			case ELEMENT_TYPE_UINT32:
				return sizeof(uint32_t);
				break;
			case ELEMENT_TYPE_UINT64:
				return sizeof(uint64_t);
				break;
			case ELEMENT_TYPE_COMPLEX64:
				return sizeof(uint64_t);
				break;
			case ELEMENT_TYPE_COMPLEX128:
				return 2* sizeof(uint64_t);
				break;
			case ELEMENT_TYPE_BFLOAT16:
				return sizeof(uint16_t);
				break;
			//case ELEMENT_TYPE_FLOAT8E4M3FN:
			//	return sizeof(
			//	break;
			//case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
			//	return sizeof(
			//	break;
			//case ELEMENT_TYPE_FLOAT8E5M2:
			//	return sizeof(
			//	break;
			//case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
			//	return sizeof(
			//	break;
			default:
				return 0;
				break;
		}
		return 0;
	}
	static short GetComponentBytesFromVulkanFormat(nosFormat format)
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
		case NOS_FORMAT_R16_USCALED:
		case NOS_FORMAT_R16G16_USCALED:
		case NOS_FORMAT_R16G16B16_USCALED:
		case NOS_FORMAT_R16G16B16A16_USCALED:
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
		case NOS_FORMAT_R16_SFLOAT:
		case NOS_FORMAT_R16G16_SFLOAT:
		case NOS_FORMAT_R16G16B16_SFLOAT:
		case NOS_FORMAT_R16G16B16A16_SFLOAT:
			return 2;

		case NOS_FORMAT_R32_UINT:
		case NOS_FORMAT_R32G32_UINT:
		case NOS_FORMAT_R32G32B32_UINT:
		case NOS_FORMAT_R32G32B32A32_UINT:
		//case NOS_FORMAT_A2R10G10B10_UINT_PACK32:
		//case NOS_FORMAT_A2R10G10B10_UNORM_PACK32:
		//case NOS_FORMAT_A2R10G10B10_USCALED_PACK32:
		//case NOS_FORMAT_X8_D24_UNORM_PACK32:
		//case NOS_FORMAT_A2R10G10B10_SNORM_PACK32:
		//case NOS_FORMAT_A2R10G10B10_SINT_PACK32:
		//case NOS_FORMAT_A2R10G10B10_SSCALED_PACK32:
		//case NOS_FORMAT_B10G11R11_UFLOAT_PACK32:
		case NOS_FORMAT_R32_SINT:
		case NOS_FORMAT_R32G32_SINT:
		case NOS_FORMAT_R32G32B32_SINT:
		case NOS_FORMAT_R32G32B32A32_SINT:
		case NOS_FORMAT_R32_SFLOAT:
		case NOS_FORMAT_R32G32_SFLOAT:
		case NOS_FORMAT_R32G32B32_SFLOAT:
		case NOS_FORMAT_R32G32B32A32_SFLOAT:
		case NOS_FORMAT_D32_SFLOAT:
			return 4;
		}
		return 0;
	}
	static short GetComponentNumFromVulkanFormat(nosFormat format)
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
		case NOS_FORMAT_G8B8G8R8_422_UNORM:
		case NOS_FORMAT_B8G8R8G8_422_UNORM:
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
	static TensorElementType GetTensorElementTypeFromVulkanFormat(nosFormat format) {
		switch (format) {
		/*
		* VK_FORMAT_R8G8B8A8_SRGB specifies a four-component, 32-bit unsigned normalized format that 
		has an 8-bit R component stored with sRGB nonlinear encoding in byte 0, an 8-bit G component stored with sRGB nonlinear 
		encoding in byte 1, an 8-bit B component stored with sRGB nonlinear encoding in byte 2, and an 8-bit A component in byte 3.

		VK_FORMAT_B8G8R8A8_UNORM specifies a four-component, 32-bit unsigned normalized format that has an 8-bit B component in byte 0,
		an 8-bit G component in byte 1, an 8-bit R component in byte 2, and an 8-bit A component in byte 3.
		*/
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
			return ELEMENT_TYPE_UINT8;

		case NOS_FORMAT_R16_UNORM:
		case NOS_FORMAT_R16G16_UNORM:
		case NOS_FORMAT_R16G16B16_UNORM:
		case NOS_FORMAT_R16G16B16A16_UNORM:
		case NOS_FORMAT_D16_UNORM:
		case NOS_FORMAT_R16_UINT:
		case NOS_FORMAT_R16G16B16_UINT:
		case NOS_FORMAT_R16G16_UINT:
		case NOS_FORMAT_R16G16B16A16_UINT:
		case NOS_FORMAT_R16_USCALED:
		case NOS_FORMAT_R16G16_USCALED:
		case NOS_FORMAT_R16G16B16_USCALED:
		case NOS_FORMAT_R16G16B16A16_USCALED:
			return ELEMENT_TYPE_UINT16;

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
			return ELEMENT_TYPE_INT16;

		case NOS_FORMAT_R16_SFLOAT:
		case NOS_FORMAT_R16G16_SFLOAT:
		case NOS_FORMAT_R16G16B16_SFLOAT:
		case NOS_FORMAT_R16G16B16A16_SFLOAT:
			return ELEMENT_TYPE_FLOAT16;

		case NOS_FORMAT_R32_UINT:
		case NOS_FORMAT_R32G32_UINT:
		case NOS_FORMAT_R32G32B32_UINT:
		case NOS_FORMAT_R32G32B32A32_UINT:
			return ELEMENT_TYPE_UINT32;

		case NOS_FORMAT_R32_SINT:
		case NOS_FORMAT_R32G32_SINT:
		case NOS_FORMAT_R32G32B32_SINT:
		case NOS_FORMAT_R32G32B32A32_SINT:
			return ELEMENT_TYPE_INT32;

		case NOS_FORMAT_R32_SFLOAT:
		case NOS_FORMAT_R32G32_SFLOAT:
		case NOS_FORMAT_R32G32B32_SFLOAT:
		case NOS_FORMAT_R32G32B32A32_SFLOAT:
		case NOS_FORMAT_D32_SFLOAT:
			return ELEMENT_TYPE_FLOAT;

		default:
			return ELEMENT_TYPE_UNDEFINED;
		}
	}

	static uint64_t GetTensorSizeFromCreateInfo(nosTensorCreateInfo createInfo) {
		uint64_t totalElementCount = 1;
		for (int i = 0; i < createInfo.ShapeInfo.DimensionCount; i++) {
			totalElementCount *= createInfo.ShapeInfo.Dimensions[i];
		}
		return static_cast<uint64_t>(GetSizeOfElementType(createInfo.ElementType)) * totalElementCount;
	}

	static uint64_t GetVulkanTextureSizeLinear(nosTextureInfo texture) {
		return ((uint64_t)texture.Width * (uint64_t)texture.Height 
			* (uint64_t)nos::tensor::GetComponentBytesFromVulkanFormat(texture.Format) 
			* (uint64_t)nos::tensor::GetComponentNumFromVulkanFormat(texture.Format));
	}

}

#endif //TENSOR_SUBSYS_COMMON_H_INCLUDED
