// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include <random>

#include "Names.h"

namespace nos::utilities
{

NOS_REGISTER_NAME(Textures);
NOS_REGISTER_NAME(Texture_Count);
NOS_REGISTER_NAME(Merge_Pass);
NOS_REGISTER_NAME(Merge_Shader);
NOS_REGISTER_NAME(Background_Color);
NOS_REGISTER_NAME(PerTextureData);
NOS_REGISTER_NAME(Blends);
NOS_REGISTER_NAME(Opacities);
NOS_REGISTER_NAME_SPACED(Merge, "nos.utilities.Merge")

struct MergePin
{
	uuid TextureId;
	uuid OpacityId;
	uuid BlendId;
};

struct MergeContext : NodeContext
{
	std::vector<MergePin> AddedPins;

	static uint32_t GetTextureCount(size_t pinCount)
	{
		return (pinCount - 2) / 3;
	}

	uint32_t GetTextureCount() const
	{
		return MergeContext::GetTextureCount(Pins.size());
	}

	MergeContext(nosFbNodePtr node) : NodeContext(node) 
	{
		auto textureCount = GetTextureCount();
		if (textureCount > 2)
		{
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<PartialPinUpdate>> updatePins;

			for (int i = 0; i < node->pins()->size(); ++i)
			{
				if (node->pins()->Get(i)->orphan_state()->type() == fb::PinOrphanStateType::ORPHAN)
				{
					updatePins.emplace_back(
						nos::CreatePartialPinUpdate
						(fbb, node->pins()->Get(i)->id(),
							0, nos::fb::CreatePinOrphanStateDirect(fbb, fb::PinOrphanStateType::ACTIVE)));
				}
			}

			// Refill AddedPins
			for (int i = 2; i < textureCount; ++i)
			{
				std::string count = std::to_string(i);
				std::string texPinName = "Texture_" + count;
				std::string opacityPinName = "Opacity_" + count;
				std::string blendPinName = "Blend_Mode_" + count;

				auto texId = *GetPinId(nos::Name(texPinName));
				auto opacityId = *GetPinId(nos::Name(opacityPinName));
				auto blendId = *GetPinId(nos::Name(blendPinName));

				AddedPins.push_back(MergePin{texId, opacityId, blendId});
			}

			HandleEvent(
				CreateAppEvent(fbb,
					nos::CreatePartialNodeUpdateDirect(fbb,
						node->id(),
						ClearFlags::NONE,
						0,
						0,
						0,
						0,
						0,
						0,
						0,
						&updatePins)));
		}
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		auto values = GetPinValues(params);
		const nosResourceShareInfo output = vkss::DeserializeTextureInfo(values[NSN_Out]);

		std::vector<nosShaderBinding> bindings;
		auto textureCount = GetTextureCount(params->PinCount);
		std::vector<nosResourceShareInfo> textures(textureCount);

		std::array<int, 16> blends = {};
		std::array<float, 16> opacities = {};
		
		uint32_t curr = 0;
		
		for (size_t i = 0; i < params->PinCount; ++i)
		{
			if(NSN_Out == params->Pins[i].Name)
				continue;

			auto val = params->Pins[i].Data;
			if (NSN_Background_Color == params->Pins[i].Name)
			{
				bindings.emplace_back(nosShaderBinding{ .Name = Name(params->Pins[i].Name), .Data = val->Data, .Size = val->Size });
				continue;
			}

			std::string name = Name(params->Pins[i].Name).AsString();
			uint32_t idx = std::stoi(name.substr(name.find_last_of('_') + 1));
		
			switch (name[0])
			{
			case 'T': textures[idx] = vkss::DeserializeTextureInfo(val->Data); break;
			case 'B': blends[idx] = *(int*)val->Data; break;
			case 'O': opacities[idx] = *(float*)val->Data; break;
			}
		}

		bindings.emplace_back(vkss::ShaderBinding(NSN_Blends, blends));
		bindings.emplace_back(vkss::ShaderBinding(NSN_Opacities, opacities));
		bindings.emplace_back(vkss::ShaderBinding(NSN_Texture_Count, textureCount));
		bindings.emplace_back(vkss::ShaderBinding(NSN_Textures, textures.data(), textures.size()));

		nosRunPassParams mergePass{
			.Key = NSN_Merge_Pass,
			.Bindings = bindings.data(),
			.BindingCount = static_cast<uint32_t>(bindings.size()),
			.Output = output,
		};

		nosCmd cmd{};
		nosCmdBeginParams begin{  .Name = NSN_Merge_Pass, .AssociatedNodeId = NodeId, .OutCmdHandle = &cmd };
		nosVulkan->Begin(&begin);
		nosVulkan->RunPass(cmd, &mergePass);
		nosVulkan->End(cmd, nullptr);
		return NOS_RESULT_SUCCESS;
	}

	void OnMenuRequested(nosContextMenuRequestPtr request) override
	{
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<flatbuffers::Offset<nos::ContextMenuItem>> items = {
			nos::CreateContextMenuItemDirect(fbb, "Add Texture", 1),
		};
		if (GetTextureCount() > 2)
			items.push_back(nos::CreateContextMenuItemDirect(fbb, "Delete Last Texture", 2));

		auto event = CreateAppEvent(fbb,
		                            CreateAppContextMenuUpdate(fbb,
		                                                    &NodeId,
		                                                    request->pos(),
		                                                    request->instigator(),
		                                                    fbb.CreateVector(items)
			                            ));

		HandleEvent(event);
	}

	void OnMenuCommand(uuid const& itemID, uint32_t cmd) override
	{
		if (!cmd)
			return;

		if (cmd == 1)
		{
			auto count = std::to_string(GetTextureCount());
			nosBuffer buffer;
			nosEngine.GetDefaultValueOfType(NOS_NAME_STATIC("nos.sys.vulkan.Texture"), &buffer);

			std::string texPinName = "Texture_" + count;
			uuid texId = nosEngine.GenerateID();
			std::vector<uint8_t> texData((uint8_t*)buffer.Data, (uint8_t*)buffer.Data + buffer.Size);

			std::string opacityPinName = "Opacity_" + count;
			uuid opacityId = nosEngine.GenerateID();
			std::vector<uint8_t> opacityData = nos::Buffer::From(1.f);
			std::vector<uint8_t> opacityMinData = nos::Buffer::From(0.f);
			std::vector<uint8_t> opacityMaxData = nos::Buffer::From(1.f);

			std::string blendPinName = "Blend_Mode_" + count;
			uuid blendId = nosEngine.GenerateID();
			std::vector<uint8_t> blendModeData = nos::Buffer::From(0u);

			std::string pinCategory = "Layer (" + count + ")";

			AddedPins.push_back({texId, opacityId, blendId});

			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<nos::fb::Pin>> pins = {
				fb::CreatePinDirect(fbb,
				                    &texId,
				                    texPinName.c_str(),
				                    "nos.sys.vulkan.Texture",
				                    fb::ShowAs::INPUT_PIN,
				                    fb::CanShowAs::INPUT_PIN_ONLY,
				                    pinCategory.c_str(),
				                    0,
				                    &texData),
				fb::CreatePinDirect(fbb,
				                    &opacityId,
				                    opacityPinName.c_str(),
				                    "float",
				                    fb::ShowAs::PROPERTY,
				                    fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
				                    pinCategory.c_str(),
				                    0,
				                    &opacityData,
				                    0,
				                    &opacityMinData,
				                    &opacityMaxData),
				fb::CreatePinDirect(fbb,
				                    &blendId,
				                    blendPinName.c_str(),
				                    "nos.utilities.BlendMode",
				                    fb::ShowAs::PROPERTY,
				                    fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
				                    pinCategory.c_str(),
				                    0,
				                    &blendModeData),
			};
			HandleEvent(CreateAppEvent(fbb,
			                                    CreatePartialNodeUpdateDirect(
				                                    fbb,
				                                    &NodeId,
				                                    ClearFlags::NONE,
				                                    0,
				                                    &pins)));
		}
		if (cmd == 2)
		{
			if (GetTextureCount() > 2)
			{
				flatbuffers::FlatBufferBuilder fbb;
				std::vector<fb::UUID> ids = {AddedPins.back().TextureId, AddedPins.back().OpacityId,
				                           AddedPins.back().BlendId};
				HandleEvent(CreateAppEvent(fbb,
				                                    CreatePartialNodeUpdateDirect(
					                                    fbb,
					                                    &NodeId,
					                                    ClearFlags::NONE,
					                                    &ids)));
				AddedPins.pop_back();
			}
		}
	}
};

nosResult RegisterMerge(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NSN_Merge, MergeContext, out);

	fs::path root = nosEngine.Module->RootFolderPath;
	auto mergePath = (root / "Shaders" / "Merge.frag").generic_string();

	// Register shaders
	nosShaderInfo shader = {.ShaderName = NSN_Merge_Shader, .Source = {.Stage = NOS_SHADER_STAGE_FRAG, .GLSLPath = mergePath.c_str() }, .AssociatedNodeClassName = NSN_Merge};
	auto ret = nosVulkan->RegisterShaders(1, &shader);
	if (NOS_RESULT_SUCCESS != ret)
		return ret;

	nosPassInfo pass = {
		.Key = NSN_Merge_Pass,
		.Shader = NSN_Merge_Shader,
		.MultiSample = 1,
        .Blend = NOS_BLEND_MODE_ALPHA_BLENDING,
	};
    
	ret = nosVulkan->RegisterPasses(1, &pass);
	return ret;
}

}
