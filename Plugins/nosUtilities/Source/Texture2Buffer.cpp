// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

// External
#include <nosVulkanSubsystem/Helpers.hpp>

#include "Names.h"

namespace nos::utilities
{
NOS_REGISTER_NAME(Texture2Buffer)
NOS_REGISTER_NAME(OutputBuffer)

nosBufferElementType GetBufferElementTypeFromVulkanFormat(nosFormat format);

struct Texture2BufferNode : nos::NodeContext
{
	nosUUID InputPinId = {}, OutputBufferPinId = {};
	Texture2BufferNode(nosFbNode const* node) : NodeContext(node)
	{
		for (const auto& pin : *node->pins()) {
			const char* currentPinName = pin->name()->c_str();
			if (NSN_Input.Compare(currentPinName) == 0) {
				InputPinId = *pin->id();
			}
			else if (NSN_OutputBuffer.Compare(currentPinName) == 0) {
				OutputBufferPinId = *pin->id();
			}
		}
	}
	
	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nos::NodeExecuteParams pins(params);
		nosResourceShareInfo inputTextureDesc = nos::vkss::DeserializeTextureInfo(pins[NSN_Input].Data->Data);
		const nosBuffer* outputBufferPinData = pins[NSN_OutputBuffer].Data;
		nosResourceShareInfo outputBufferDesc = nos::vkss::ConvertToResourceInfo(*static_cast<sys::vulkan::Buffer*>(outputBufferPinData->Data));
		uint64_t currentSize = inputTextureDesc.Memory.Size;
		
		if (currentSize != outputBufferDesc.Memory.Size) {
			// Need to create a new buffer
			outputBufferDesc.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			outputBufferDesc.Info.Buffer.Size = currentSize;
			outputBufferDesc.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_SRC);
			outputBufferDesc.Info.Buffer.ElementType = GetBufferElementTypeFromVulkanFormat(inputTextureDesc.Info.Texture.Format);
		}

		// Create buffer or update field type of existing buffer
		outputBufferDesc.Info.Buffer.FieldType = inputTextureDesc.Info.Texture.FieldType;
		nosEngine.SetPinValue(OutputBufferPinId, nos::Buffer::From(vkss::ConvertBufferInfo(outputBufferDesc)));

		outputBufferDesc = nos::vkss::ConvertToResourceInfo(*static_cast<sys::vulkan::Buffer*>(outputBufferPinData->Data));
		nosCmd cmd = {};
		nosCmdBeginParams beginParams = {.Name = NOS_NAME("Texture2Buffer Copy"), .AssociatedNodeId = NodeId, .OutCmdHandle = &cmd};
		nosVulkan->Begin2(&beginParams);
		nosVulkan->Copy(cmd, &inputTextureDesc, &outputBufferDesc, 0);
		nosVulkan->End(cmd, nullptr);
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterTexture2Buffer(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_Texture2Buffer, Texture2BufferNode, fn);
	return NOS_RESULT_SUCCESS;
}

// TODO: Maybe move it to nos.sys.vulkan subsystem
nosBufferElementType GetBufferElementTypeFromVulkanFormat(nosFormat format) {
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
		return NOS_BUFFER_ELEMENT_TYPE_UINT8;

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
		return NOS_BUFFER_ELEMENT_TYPE_UINT16;

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
		return NOS_BUFFER_ELEMENT_TYPE_INT16;

	case NOS_FORMAT_R16_SFLOAT:
	case NOS_FORMAT_R16G16_SFLOAT:
	case NOS_FORMAT_R16G16B16_SFLOAT:
	case NOS_FORMAT_R16G16B16A16_SFLOAT:
		return NOS_BUFFER_ELEMENT_TYPE_FLOAT16;

	case NOS_FORMAT_R32_UINT:
	case NOS_FORMAT_R32G32_UINT:
	case NOS_FORMAT_R32G32B32_UINT:
	case NOS_FORMAT_R32G32B32A32_UINT:
		return NOS_BUFFER_ELEMENT_TYPE_UINT32;

	case NOS_FORMAT_R32_SINT:
	case NOS_FORMAT_R32G32_SINT:
	case NOS_FORMAT_R32G32B32_SINT:
	case NOS_FORMAT_R32G32B32A32_SINT:
		return NOS_BUFFER_ELEMENT_TYPE_INT32;

	case NOS_FORMAT_R32_SFLOAT:
	case NOS_FORMAT_R32G32_SFLOAT:
	case NOS_FORMAT_R32G32B32_SFLOAT:
	case NOS_FORMAT_R32G32B32A32_SFLOAT:
	case NOS_FORMAT_D32_SFLOAT:
		return NOS_BUFFER_ELEMENT_TYPE_FLOAT;

	default:
		return NOS_BUFFER_ELEMENT_TYPE_UNDEFINED;
	}
}
}
