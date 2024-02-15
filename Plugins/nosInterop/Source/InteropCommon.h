#pragma once
#ifndef INTEROP_COMMON_H_INCLUDED
#define INTEROP_COMMON_H_INCLUDED
#include "nosVulkanSubsystem/Types_generated.h"
#include "nosCUDASubsystem/Types_generated.h"
#include "Nodos/PluginAPI.h"
#include "nosVulkanSubsystem/nosVulkanSubsystem.h"

union ElementType {
	nos::sys::vulkan::BufferElementType VulkanElementType;
	nos::sys::cuda::BufferElementType CUDAElementType;
};

typedef struct BufferPin {
	ElementType Element;
	uint64_t Size;
	uint64_t Address;
	uint64_t Offset;
};

typedef struct PinConfig {
	const char* Name;
	nos::fb::ShowAs ShowAs;
	nos::fb::CanShowAs CanShowAs;
}PinConfig;

__forceinline void CreateOrUpdateVulkanBufferPin(BufferPin bufferPin, nosUUID* NodeUUID, nosUUID* GeneratedPinUUID, PinConfig config) {
	std::vector<nos::fb::UUID> pinsToDelete = { *GeneratedPinUUID };
	flatbuffers::FlatBufferBuilder fbb;
	std::vector<flatbuffers::Offset<nos::app::AppEvent>> Offsets;
	auto deletePinEvent = nos::CreateAppEvent(fbb,
		nos::CreatePartialNodeUpdateDirect(fbb, NodeUUID, nos::ClearFlags::NONE, &pinsToDelete));
	nosResult res = nosEngine.EnqueueEvent(&deletePinEvent);

	flatbuffers::FlatBufferBuilder fbb2;
	nos::sys::vulkan::Buffer buffer;
	buffer.mutate_element_type(bufferPin.Element.VulkanElementType);
	buffer.mutate_handle(bufferPin.Address);
	buffer.mutate_size_in_bytes(bufferPin.Size);
	buffer.mutate_offset(bufferPin.Offset);
	
	auto bufPin = nos::Buffer::From(buffer);
	auto bufferPinBytes = std::vector<uint8_t>((uint8_t*)bufPin.Data(), (uint8_t*)bufPin.Data() + bufPin.Size());
	nosEngine.GenerateID(GeneratedPinUUID);
	std::vector<flatbuffers::Offset<nos::fb::Pin>> Pins;
	Pins.push_back(nos::fb::CreatePinDirect(fbb2,
		GeneratedPinUUID,
		config.Name,
		nos::sys::vulkan::Buffer::GetFullyQualifiedName(),
		config.ShowAs,
		config.CanShowAs,
		0,
		0,
		&bufferPinBytes));
	auto createPinEvent = CreateAppEvent(fbb2,
		nos::CreatePartialNodeUpdateDirect(fbb2, NodeUUID, nos::ClearFlags::NONE, 0, &Pins));
	res = nosEngine.EnqueueEvent(&createPinEvent);
}

__forceinline void CreateOrUpdateCUDABufferPin(BufferPin bufferPin, nosUUID* NodeUUID, nosUUID* GeneratedPinUUID, PinConfig config) {
	std::vector<nos::fb::UUID> pinsToDelete = { *GeneratedPinUUID };
	flatbuffers::FlatBufferBuilder fbb;
	std::vector<flatbuffers::Offset<nos::app::AppEvent>> Offsets;
	auto deletePinEvent = nos::CreateAppEvent(fbb,
		nos::CreatePartialNodeUpdateDirect(fbb, NodeUUID, nos::ClearFlags::NONE, &pinsToDelete));
	nosResult res = nosEngine.EnqueueEvent(&deletePinEvent);

	flatbuffers::FlatBufferBuilder fbb2;
	nos::sys::cuda::Buffer buffer;
	buffer.mutate_element_type(bufferPin.Element.CUDAElementType);
	buffer.mutate_handle(bufferPin.Address);
	buffer.mutate_size_in_bytes(bufferPin.Size);
	buffer.mutate_offset(bufferPin.Offset);

	auto bufPin = nos::Buffer::From(buffer);
	auto bufferPinBytes = std::vector<uint8_t>((uint8_t*)bufPin.Data(), (uint8_t*)bufPin.Data() + bufPin.Size());
	nosEngine.GenerateID(GeneratedPinUUID);
	std::vector<flatbuffers::Offset<nos::fb::Pin>> Pins;
	Pins.push_back(nos::fb::CreatePinDirect(fbb2,
		GeneratedPinUUID,
		config.Name,
		nos::sys::vulkan::Buffer::GetFullyQualifiedName(),
		config.ShowAs,
		config.CanShowAs,
		0,
		0,
		&bufferPinBytes));
	auto createPinEvent = CreateAppEvent(fbb2,
		nos::CreatePartialNodeUpdateDirect(fbb2, NodeUUID, nos::ClearFlags::NONE, 0, &Pins));
	res = nosEngine.EnqueueEvent(&createPinEvent);
}

__forceinline short GetComponentBytesFromVulkanFormat(nosFormat format)
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

__forceinline short GetComponentNumFromVulkanFormat(nosFormat format)
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

__forceinline int GetBufferElementTypeFromVulkanFormat(nosFormat format) {
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
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_UINT8;

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
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_UINT16;

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
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_INT16;

	case NOS_FORMAT_R16_SFLOAT:
	case NOS_FORMAT_R16G16_SFLOAT:
	case NOS_FORMAT_R16G16B16_SFLOAT:
	case NOS_FORMAT_R16G16B16A16_SFLOAT:
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_FLOAT16;

	case NOS_FORMAT_R32_UINT:
	case NOS_FORMAT_R32G32_UINT:
	case NOS_FORMAT_R32G32B32_UINT:
	case NOS_FORMAT_R32G32B32A32_UINT:
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_UINT32;

	case NOS_FORMAT_R32_SINT:
	case NOS_FORMAT_R32G32_SINT:
	case NOS_FORMAT_R32G32B32_SINT:
	case NOS_FORMAT_R32G32B32A32_SINT:
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_INT32;

	case NOS_FORMAT_R32_SFLOAT:
	case NOS_FORMAT_R32G32_SFLOAT:
	case NOS_FORMAT_R32G32B32_SFLOAT:
	case NOS_FORMAT_R32G32B32A32_SFLOAT:
	case NOS_FORMAT_D32_SFLOAT:
		return (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_FLOAT;

	default:
		return  (int)nos::sys::cuda::BufferElementType::ELEMENT_TYPE_UNDEFINED;
	}
}

#endif //INTEROP_COMMON_H_INCLUDED