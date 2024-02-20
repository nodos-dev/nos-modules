#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>
#include <nosVulkanSubsystem/Types_generated.h>
#include <nosCUDASubsystem/nosCUDASubsystem.h>
#include <nosCUDASubsystem/Types_generated.h>
#include "InteropCommon.h"
#include "InteropNames.h"

struct TextureFormatConverter : nos::NodeContext
{
	BufferPin VulkanBufferPinProxy = {};
	nosResourceShareInfo InputTexture = {}, OutputTexture = {};
	nosUUID NodeUUID = {}, InputUUID = {}, OutputUUID = {}, FormatUUID = {};
	nos::sys::vulkan::Format OutputFormat = {};
	TextureFormatConverter(nosFbNode const* node) : NodeContext(node)
	{
		NodeUUID = *node->id();

		for (const auto& pin : *node->pins()) {
			if (NSN_Input.Compare(pin->name()->c_str()) == 0) {
				InputUUID = *pin->id();
			}
			else if (NSN_Output.Compare(pin->name()->c_str()) == 0) {
				OutputUUID = *pin->id();
			}
		}
		std::vector<std::string> Formats = {};
		auto FormatEnums = nos::sys::vulkan::EnumValuesFormat();
		for (int i = 0; i < 69; i++) {
			Formats.push_back(nos::sys::vulkan::EnumNameFormat(FormatEnums[i]));
		}
		CreateStringList(FormatUUID, NodeUUID, "OutputFormat", std::move(Formats));
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (InputUUID == pinId) {
			InputTexture = nos::vkss::DeserializeTextureInfo(value.Data);
		}
		if (FormatUUID == pinId) {
			const char* SelectedFormat = (const char*)value.Data;
			auto FormatEnums = nos::sys::vulkan::EnumValuesFormat();
			nos::sys::vulkan::Format newFormat = {};
			for (int i = 0; i < 69; i++) {
				if (strcmp(nos::sys::vulkan::EnumNameFormat(FormatEnums[i]), SelectedFormat) == 0) {
					newFormat = FormatEnums[i];
					if (newFormat != OutputFormat) {
						OutputFormat = newFormat;
						PrepareResources();
					}
					break;
				}
			}
		}
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		auto pinIds = nos::GetPinIds(args);
		auto pinValues = nos::GetPinValues(args);
		InputTexture = nos::vkss::DeserializeTextureInfo(pinValues[NSN_Input]);
		auto Out = nos::vkss::DeserializeTextureInfo(pinValues[NSN_Output]);
		nosCmd cmd = {};
		nosGPUEvent waitEvent = {};
		nosCmdEndParams endParams = { .ForceSubmit = true, .OutGPUEventHandle = &waitEvent };
		nosVulkan->Begin("TexToTex", &cmd);
		nosVulkan->Copy(cmd, &InputTexture, &Out, nullptr);
		nosVulkan->End(cmd, &endParams);
		nosVulkan->WaitGpuEvent(&waitEvent, UINT64_MAX);
		
		//nosResourceShareInfo out = nos::vkss::DeserializeTextureInfo(pinValues[NSN_Output]);
		//nosCmd cmd2;
		//nosGPUEvent gpuevent2 = {};
		//nosCmdEndParams endParams2 = { .ForceSubmit = true, .OutGPUEventHandle = &gpuevent2 };
		//nosVulkan->Begin("NVVFX Upload", &cmd2);
		//nosVulkan->Copy(cmd2, &OutputTexture, &out, 0);
		//nosVulkan->End(cmd2, &endParams2);
		//nosVulkan->WaitGpuEvent(&gpuevent2, UINT64_MAX);

		return NOS_RESULT_SUCCESS;
	}

	void PrepareResources() {
		if (OutputTexture.Memory.Handle != NULL) {
			//nosVulkan->DestroyResource(&OutputTexture);
		}

		OutputTexture.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		OutputTexture.Info.Texture.FieldType = InputTexture.Info.Texture.FieldType;
		OutputTexture.Info.Texture.Filter = InputTexture.Info.Texture.Filter;
		OutputTexture.Info.Texture.Format = nosFormat((int)OutputFormat);
		OutputTexture.Info.Texture.Height = InputTexture.Info.Texture.Height;
		OutputTexture.Info.Texture.Usage = InputTexture.Info.Texture.Usage;
		OutputTexture.Info.Texture.Width = InputTexture.Info.Texture.Width;

		//nosVulkan->CreateResource(&OutputTexture);
		auto TTexture = nos::vkss::ConvertTextureInfo(OutputTexture);
		
		nosEngine.SetPinValue(OutputUUID, nos::Buffer::From(TTexture));
	}
	


};

nosResult RegisterTextureFormatConverter(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_TextureFormatConverter, TextureFormatConverter, fn);
	return NOS_RESULT_SUCCESS;
}

