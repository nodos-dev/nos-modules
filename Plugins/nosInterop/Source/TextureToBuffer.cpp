#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosVulkanSubsystem/Types_generated.h>
#include "InteropCommon.h"
#include "InteropNames.h"

	
struct TextureToBufferNodeContext : nos::NodeContext
{
	BufferPin BufferPinProxy = {};
	nosResourceShareInfo Buffer = {};
	nosUUID NodeUUID = {}, InputUUID = {}, OutputBufferUUID = {};
	TextureToBufferNodeContext(nosFbNode const* node) : NodeContext(node)
	{
		NodeUUID = *node->id();

		for (const auto& pin : *node->pins()) {
			const char* currentPinName = pin->name()->c_str();
			if (NSN_Input.Compare(pin->name()->c_str()) == 0) {
				InputUUID = *pin->id();
			}
			else if (NSN_OutputBuffer.Compare(pin->name()->c_str()) == 0) {
				OutputBufferUUID = *pin->id();
			}
		}
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		auto pinIds = nos::GetPinIds(args);
		auto pinValues = nos::GetPinValues(args);
		nosResourceShareInfo inputTextureInfo = nos::vkss::DeserializeTextureInfo(pinValues[NSN_Input]);
		PrepareResources(inputTextureInfo);
		return NOS_RESULT_SUCCESS;
	}

	void PrepareResources(nosResourceShareInfo& in) {
		uint64_t currentSize = static_cast<uint64_t>(GetComponentBytesFromVulkanFormat(in.Info.Texture.Format)) *
			static_cast<uint64_t>(GetComponentNumFromVulkanFormat(in.Info.Texture.Format)) * in.Info.Texture.Width * in.Info.Texture.Height;
		
		if (currentSize == Buffer.Memory.Size) {
			nosCmd texToBuf = {};
			nosGPUEvent waitTexToBuf = {};
			nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitTexToBuf };
			nosVulkan->Begin("TexToBuf", &texToBuf);
			nosVulkan->Copy(texToBuf, &in, &Buffer, 0);
			nosVulkan->End(texToBuf, &endParams);
			nosVulkan->WaitGpuEvent(&waitTexToBuf, UINT64_MAX);
			return;
		}

		if (Buffer.Memory.Handle != NULL) {
			nosVulkan->DestroyResource(&Buffer);
		}
		Buffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		Buffer.Info.Buffer.Size = currentSize;
		Buffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_SRC);
		nosVulkan->CreateResource(&Buffer);

		nos::sys::vulkan::Buffer buffer;
		buffer.mutate_element_type((nos::sys::vulkan::BufferElementType)GetBufferElementTypeFromVulkanFormat(in.Info.Texture.Format));
		buffer.mutate_handle(Buffer.Memory.Handle);
		buffer.mutate_size_in_bytes(Buffer.Memory.Size);
		buffer.mutate_offset(Buffer.Memory.ExternalMemory.Offset);

		buffer.mutable_external_memory().mutate_allocation_size(Buffer.Memory.ExternalMemory.AllocationSize);
		buffer.mutable_external_memory().mutate_handle(Buffer.Memory.ExternalMemory.Handle);
		buffer.mutable_external_memory().mutate_handle_type(Buffer.Memory.ExternalMemory.HandleType);
		buffer.mutable_external_memory().mutate_pid(Buffer.Memory.ExternalMemory.PID);



		nos::Buffer bufPin = nos::Buffer::From(buffer);

		nosEngine.SetPinValueDirect(OutputBufferUUID, bufPin);
		
		return;
	}

};

nosResult RegisterTextureToBuffer(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_TextureToBuffer, TextureToBufferNodeContext, fn);
	return NOS_RESULT_SUCCESS;
}

