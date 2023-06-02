// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>
#include <MediaZ/PluginAPI.h>
#include "Merge.frag.spv.dat"

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
MZ_REGISTER_NAME2(Out);
MZ_REGISTER_NAME2(Texture_Count);
MZ_REGISTER_NAME2(Merge_Pass);
MZ_REGISTER_NAME2(Merge_Shader);

struct MergeContext
{
	mzUUID Id;
	uint32_t TextureCount = 2;

	void Run(const mzNodeExecuteArgs* pins)
	{
		auto values = GetPinValues(pins);
		const mzResourceShareInfo output = DeserializeTextureInfo(values[Out_Name]);

		std::vector<mzShaderBinding> bindings;
		std::vector<mzResourceShareInfo> Textures(TextureCount);
		u32 curr = 0;

		for (size_t i = 0; i < pins->PinCount; ++i)
		{
			std::string name = Name(pins->PinNames[i]).AsString();
			mzBuffer val = pins->PinValues[i];
			if (name.starts_with("Texture_"))
			{
				bindings.emplace_back(ShaderBinding(Name(pins->PinNames[i]), Textures[curr++] = DeserializeTextureInfo(val.Data)));
			}
			else if (name != "Out")
				bindings.emplace_back(mzShaderBinding{.Name = Name(pins->PinNames[i]), .FixedSize = val.Data});
		}

		bindings.emplace_back(ShaderBinding(Texture_Count_Name, TextureCount));

		mzRunPassParams mergePass{
			.Key = Merge_Pass_Name,
			.Bindings = bindings.data(),
			.BindingCount = static_cast<uint32_t>(bindings.size()),
			.Output = output,
		};

		mzEngine.RunPass(nullptr, &mergePass);
	}

	static void OnMenuRequested(void* ctx, const mzContextMenuRequest* request)
	{
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<flatbuffers::Offset<mz::ContextMenuItem>> items = {
			mz::CreateContextMenuItemDirect(fbb, "Add Input Texture", 1)
		};

		auto event = CreateAppEvent(fbb,
		                             CreateContextMenuUpdate(fbb,
		                                                         &static_cast<MergeContext*>(ctx)->Id,
		                                                         request->pos(),
		                                                         request->instigator(), fbb.CreateVector(items)
		                                                         ));

		
		mzEngine.HandleEvent(event);
	}

	static void OnNodeCreated(const mzFbNode* node, void** outCtxPtr)
	{
		*outCtxPtr = new MergeContext();
		static_cast<MergeContext*>(*outCtxPtr)->Id = *node->id();
	}
	
	static void OnMenuCommand(void* ctx, uint32_t cmd)
	{
		if(!cmd)
			return;

		auto c = static_cast<MergeContext*>(ctx);
		auto count = std::to_string(c->TextureCount++);
		mzBuffer buffer;
		mzEngine.GetDefaultValueOfType("mz.fb.Texture", &buffer);

		std::string texPinName = "Texture_" + count;
		mzUUID texId = *(mzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> texData((u8*)buffer.Data, (u8*)buffer.Data + buffer.Size);

		std::string opacityPinName = "Opacity_" + count;
		mzUUID opacityId = *(mzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> opacityData = mz::Buffer::From(1.f);

		std::string blendPinName = "Blend_" + count;
		mzUUID blendId = *(mzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> blendModeData = mz::Buffer::From(0u);

		std::string pinCategory = "Layer (" + count + ")";

		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<mz::fb::Pin>> pins = {
			fb::CreatePinDirect(fbb, &texId, texPinName.c_str() ,"mz.fb.Texture", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_ONLY, pinCategory.c_str(),0,&texData),
			fb::CreatePinDirect(fbb, &opacityId, opacityPinName.c_str(), "float", fb::ShowAs::PROPERTY, fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY, pinCategory.c_str(),0,&opacityData),
			fb::CreatePinDirect(fbb, &blendId, blendPinName.c_str(), "mz.fb.BlendMode", fb::ShowAs::PROPERTY, fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY, pinCategory.c_str(),0,&blendModeData),
		};
		mzEngine.HandleEvent(CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &((MergeContext*)(ctx))->Id, ClearFlags::NONE,0,&pins)));
	}

	static mzResult GetShaders(size_t* outCount, mzName* outShaderNames, mzBuffer* outSpirvBufs)
	{
		*outCount = 1;
		if (!outShaderNames || !outSpirvBufs)
			return MZ_RESULT_SUCCESS;
		*outShaderNames = Merge_Shader_Name;
		outSpirvBufs->Data = (void*)(Merge_frag_spv);
		outSpirvBufs->Size = sizeof(Merge_frag_spv);
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* outCount,  mzPassInfo* passes)
	{
		*outCount = 1;
		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes = {
			.Key = Merge_Pass_Name,
			.Shader = Merge_Shader_Name,
			.Blend = 1,
			.MultiSample = 1,
		};
		return MZ_RESULT_SUCCESS;
	}

	static void OnNodeDeleted(void* ctx, mzUUID id)
	{
		delete static_cast<MergeContext*>(ctx);
	}
};

void RegisterMerge(mzNodeFunctions* out)
{
	out->TypeName = "mz.utilities.Merge";
	out->OnMenuRequested = MergeContext::OnMenuRequested;
	out->OnNodeCreated = MergeContext::OnNodeCreated;
	out->OnMenuCommand = MergeContext::OnMenuCommand;
	out->GetShaders = MergeContext::GetShaders;
	out->GetPasses = MergeContext::GetPasses;
	out->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) {
		static_cast<mz::utilities::MergeContext*>(ctx)->Run(args);
		return MZ_RESULT_SUCCESS;
	};
	out->OnNodeDeleted = MergeContext::OnNodeDeleted;
}

}
