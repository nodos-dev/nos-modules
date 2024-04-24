#include <Nodos/PluginHelpers.hpp>

// External
#include <glm/glm.hpp> // TODO: Ring no longer needs glm::mat4 colormatrix. Remove this
#include <nosVulkanSubsystem/Helpers.hpp>

namespace nos::utilities
{
struct UploadBufferNodeContext : NodeContext
{
	UploadBufferNodeContext(const nosFbNode* node) : NodeContext(node)
	{

	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		auto execArgs = nos::NodeExecuteArgs(args);
		auto& output = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Output")].Data->Data);
		auto& input = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Input")].Data->Data);

		if (input.size_in_bytes() != output.size_in_bytes())
		{
			nosResourceShareInfo bufInfo = {
				.Info = {.Type = NOS_RESOURCE_TYPE_BUFFER,
						 .Buffer = nosBufferInfo{.Size = (uint32_t)input.size_in_bytes(),
												 .Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_DST |
																		 NOS_BUFFER_USAGE_TRANSFER_SRC |
																		 NOS_BUFFER_USAGE_STORAGE_BUFFER),
												 .MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_DEVICE_MEMORY)}}};
			auto bufferDesc = vkss::ConvertBufferInfo(bufInfo);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(bufferDesc));

			output = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Output")].Data->Data);
		}

		if (!output.handle() || !input.handle())
		{
			return NOS_RESULT_SUCCESS;
		}

		auto OutputBuffer = vkss::ConvertToResourceInfo(output);
		auto InputBuffer = vkss::ConvertToResourceInfo(input);

		nosCmd cmd;
		nosVulkan->Begin("UploadBuffer Staging Copy", &cmd);
		nosVulkan->Copy(cmd, &InputBuffer, &OutputBuffer, 0);
		nosVulkan->End(cmd, nullptr);

		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterUploadBuffer(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("UploadBuffer"), UploadBufferNodeContext, functions)
	return NOS_RESULT_SUCCESS;
}

struct Buffer2TextureNodeContext : NodeContext
{
	Buffer2TextureNodeContext(const nosFbNode* node) : NodeContext(node)
	{
	}
	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs execArgs(args);
		const auto& inputPinData = *InterpretPinValue<sys::vulkan::Buffer>(execArgs[NOS_NAME_STATIC("Input")].Data->Data);
		const nosBuffer* outputPinData = execArgs[NOS_NAME_STATIC("Output")].Data;
		const auto& output = *InterpretPinValue<sys::vulkan::Texture>(outputPinData->Data);
		const auto& size = *InterpretPinValue<fb::vec2u>(execArgs[NOS_NAME_STATIC("Size")].Data->Data);
		const auto& format = *InterpretPinValue<sys::vulkan::Format>(execArgs[NOS_NAME_STATIC("Format")].Data->Data);
		if (size.x() != output.width() ||
			size.y() != output.height() ||
			format != output.format())
		{
			nosResourceShareInfo tex{.Info = {
				.Type = NOS_RESOURCE_TYPE_TEXTURE,
				.Texture = {
					.Width = size.x(),
					.Height = size.y(),
					.Format = nosFormat(format)
				}
			}};
			sys::vulkan::TTexture texDef = vkss::ConvertTextureInfo(tex);
			nosEngine.SetPinValueByName(NodeId, NOS_NAME_STATIC("Output"), Buffer::From(texDef));
		}
		nosResourceShareInfo out = vkss::DeserializeTextureInfo(outputPinData->Data);
		nosResourceShareInfo in = vkss::ConvertToResourceInfo(inputPinData);

		if (!in.Memory.Handle || !out.Memory.Handle)
			return NOS_RESULT_SUCCESS;

		nosCmd cmd;
		nosVulkan->Begin("Buffer2Texture Copy", &cmd);
		nosVulkan->Copy(cmd, &in, &out, 0);
		nosGPUEvent event;
		nosCmdEndParams params{.ForceSubmit = true, .OutGPUEventHandle = &event};
		nosVulkan->End(cmd, &params);
		nosVulkan->WaitGpuEvent(&event, UINT_MAX);
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterBuffer2Texture(nosNodeFunctions* funcs)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.utilities.Buffer2Texture"), Buffer2TextureNodeContext, funcs);
	return NOS_RESULT_SUCCESS;
}


}