// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>
#include <MediaZ/PluginAPI.h>
#include "../Shaders/Merge.frag.spv.dat"

#include <random>

namespace mz
{
std::seed_seq Seed()
{
	std::random_device rd;
	auto seed_data = std::array<int, std::mt19937::state_size>{};
	std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
	return std::seed_seq(std::begin(seed_data), std::end(seed_data));
}

std::seed_seq seed = Seed();
std::mt19937 mtengine(seed);
uuids::uuid_random_generator generator(mtengine);

}

namespace mz::utilities
{
MZ_REGISTER_NAME(Out);
MZ_REGISTER_NAME(Textures);
MZ_REGISTER_NAME(Texture_Count);
MZ_REGISTER_NAME(Merge_Pass);
MZ_REGISTER_NAME(Merge_Shader);
MZ_REGISTER_NAME(Background_Color);
MZ_REGISTER_NAME(PerTextureData);
MZ_REGISTER_NAME(Blends);
MZ_REGISTER_NAME(Opacities);
MZ_REGISTER_NAME_SPACED(Merge, "mz.utilities.Merge")

struct MergePin
{
	mzUUID TextureId;
	mzUUID OpacityId;
	mzUUID BlendId;
};

struct MergeContext : NodeContext
{
	uint32_t TextureCount = 2;
	std::vector<MergePin> AddedPins;

	MergeContext(mzFbNode const* node) : NodeContext(node) 
	{
		TextureCount = (node->pins()->size() - 2) / 3;

		if (TextureCount > 2)
		{
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<PartialPinUpdate>> updatePins;

			for (int i = 0; i < node->pins()->size(); ++i)
			{
				if (node->pins()->Get(i)->orphan_state()->is_orphan())
				{
					updatePins.emplace_back(
						mz::CreatePartialPinUpdate
						(fbb, node->pins()->Get(i)->id(),
							0, mz::fb::CreateOrphanStateDirect(fbb, false), Action::NOP));
				}
			}

			HandleEvent(
				CreateAppEvent(fbb,
					mz::CreatePartialNodeUpdateDirect(fbb,
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

	mzResult ExecuteNode(const mzNodeExecuteArgs* args) override
	{
		auto values = GetPinValues(args);
		const mzResourceShareInfo output = DeserializeTextureInfo(values[MZN_Out]);

		std::vector<mzShaderBinding> bindings;
		std::vector<mzResourceShareInfo> textures(TextureCount);

		std::array<int, 16> blends = {};
		std::array<float, 16> opacities = {};
		
		u32 curr = 0;
		
		for (size_t i = 0; i < args->PinCount; ++i)
		{
			if(MZN_Out == args->PinNames[i])
				continue;

			mzBuffer val = args->PinValues[i];
			if (MZN_Background_Color == args->PinNames[i])
			{
				bindings.emplace_back(mzShaderBinding{ .Name = Name(args->PinNames[i]), .Data = val.Data, .Size = val.Size });
				continue;
			}

			std::string name = Name(args->PinNames[i]).AsString();
			uint32_t idx = std::stoi(name.substr(name.find_last_of('_') + 1));
		
			switch (name[0])
			{
			case 'T': textures[idx] = DeserializeTextureInfo(val.Data); break;
			case 'B': blends[idx] = *(int*)val.Data; break;
			case 'O': opacities[idx] = *(float*)val.Data; break;
			}
		}

		bindings.emplace_back(ShaderBinding(MZN_Blends, blends));
		bindings.emplace_back(ShaderBinding(MZN_Opacities, opacities));
		bindings.emplace_back(ShaderBinding(MZN_Texture_Count, TextureCount));
		bindings.emplace_back(ShaderBinding(MZN_Textures, textures.data(), textures.size()));

		mzRunPassParams mergePass{
			.Key = MZN_Merge_Pass,
			.Bindings = bindings.data(),
			.BindingCount = static_cast<uint32_t>(bindings.size()),
			.Output = output,
		};

		mzEngine.RunPass(nullptr, &mergePass);
		return MZ_RESULT_SUCCESS;
	}

	void OnMenuRequested(const mzContextMenuRequest* request) override
	{
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<flatbuffers::Offset<mz::ContextMenuItem>> items = {
			mz::CreateContextMenuItemDirect(fbb, "Add Texture", 1),
		};
		if (TextureCount > 2)
			items.push_back(mz::CreateContextMenuItemDirect(fbb, "Delete Last Texture", 2));

		auto event = CreateAppEvent(fbb,
		                            CreateContextMenuUpdate(fbb,
		                                                    &NodeId,
		                                                    request->pos(),
		                                                    request->instigator(),
		                                                    fbb.CreateVector(items)
			                            ));

		HandleEvent(event);
	}

	void OnMenuCommand(mzUUID itemID, uint32_t cmd) override
	{
		if (!cmd)
			return;

		if (cmd == 1)
		{
			auto count = std::to_string(TextureCount++);
			mzBuffer buffer;
			mzEngine.GetDefaultValueOfType(MZ_NAME_STATIC("mz.fb.Texture"), &buffer);

			std::string texPinName = "Texture_" + count;
			mzUUID texId = *(mzUUID*)mz::generator().as_bytes().data();
			std::vector<uint8_t> texData((u8*)buffer.Data, (u8*)buffer.Data + buffer.Size);

			std::string opacityPinName = "Opacity_" + count;
			mzUUID opacityId = *(mzUUID*)mz::generator().as_bytes().data();
			std::vector<uint8_t> opacityData = mz::Buffer::From(1.f);
			std::vector<uint8_t> opacityMinData = mz::Buffer::From(0.f);
			std::vector<uint8_t> opacityMaxData = mz::Buffer::From(1.f);

			std::string blendPinName = "Blend_Mode_" + count;
			mzUUID blendId = *(mzUUID*)mz::generator().as_bytes().data();
			std::vector<uint8_t> blendModeData = mz::Buffer::From(0u);

			std::string pinCategory = "Layer (" + count + ")";

			AddedPins.push_back({texId, opacityId, blendId});

			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<mz::fb::Pin>> pins = {
				fb::CreatePinDirect(fbb,
				                    &texId,
				                    texPinName.c_str(),
				                    "mz.fb.Texture",
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
				                    "mz.fb.BlendMode",
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
			if (TextureCount > 2)
			{
				TextureCount--;
				flatbuffers::FlatBufferBuilder fbb;
				std::vector<mzUUID> ids = {AddedPins.back().TextureId, AddedPins.back().OpacityId,
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
    
	inline static mzShaderSource MergeSource = { .SpirvBlob = {(void*)Merge_frag_spv, sizeof(Merge_frag_spv) } };
    

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;
		outShaders[0] = {.Key = MZN_Merge_Shader, .Source = MergeSource };
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* outCount, mzPassInfo* passes)
	{
		*outCount = 1;
		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes = {
			.Key = MZN_Merge_Pass,
			.Shader = MZN_Merge_Shader,
			.Blend = 1,
			.MultiSample = 1,
		};
		return MZ_RESULT_SUCCESS;
	}
};

void RegisterMerge(mzNodeFunctions* out)
{
	MZ_BIND_NODE_CLASS(MZN_Merge, MergeContext, out);
}

}
