// Copyright MediaZ AS. All Rights Reserved.

#include "Merge.hpp"
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

namespace mz::filters
{

struct MergeContext
{
	MzUUID Id;
	uint32_t TextureCount = 0;

	void Run(const MzNodeExecuteArgs* pins)
	{
		auto values = GetPinValues(pins);
		const MzResourceShareInfo output = ValAsTex(values["Out"]);

		std::vector<MzShaderBinding> bindings;

		for (size_t i = 0; i < pins->PinCount; ++i)
		{
			std::string name = pins->PinNames[i];
			MzBuffer val = pins->PinValues[i];
			if (name.starts_with("Texture_"))
			{
				auto tex = ValAsTex(val.Data);
				bindings.emplace_back(ShaderBinding(name.c_str(), tex));
				++TextureCount;
			}
			else if (name != "Out")
				bindings.emplace_back(ShaderBinding(name.c_str(), val.Data));
		}

		bindings.emplace_back(ShaderBinding("Texture_Count", TextureCount));

		MzRunPassParams mergePass{
			.PassKey = "Merge_Pass",
			.Bindings = bindings.data(),
			.BindingCount = (uint32_t)bindings.size(),
			.Output = output,
		};

		mzEngine.RunPass(nullptr, &mergePass);
	}

	static void OnMenuRequested(void* ctx, const MzContextMenuRequest* request)
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

	static void OnNodeCreated(const MzFbNode* node, void** outCtxPtr)
	{
		*outCtxPtr = new MergeContext();
		static_cast<MergeContext*>(*outCtxPtr)->Id = *node->id();
	}
	
	static void OnMenuCommand(void* ctx, uint32_t cmd)
	{
		if(!cmd)
			return;
		MzBuffer buffer;
		mzEngine.GetDefaultValueOfType("mz.fb.Texture", &buffer);

		std::string texPinName = "Texture_" + std::to_string(static_cast<MergeContext*>(ctx)->TextureCount);
		MzUUID texId = *(MzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> texData(buffer.Size, 0);
		memcpy(texData.data(), buffer.Data, buffer.Size);

		std::string opacityPinName = "Opacity_" + std::to_string(static_cast<MergeContext*>(ctx)->TextureCount);
		MzUUID opacityId = *(MzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> opacityData(sizeof(float), 1.0f);

		std::string blendPinName = "Blend_" + std::to_string(static_cast<MergeContext*>(ctx)->TextureCount);
		MzUUID blendId = *(MzUUID*)mz::generator().as_bytes().data();
		std::vector<uint8_t> blendModeData(sizeof(unsigned int), 0);

		std::string pinCategory = "Layer (" + std::to_string(static_cast<MergeContext*>(ctx)->TextureCount) + ")";

		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<mz::fb::Pin>> pins = {
			fb::CreatePinDirect(fbb, &texId, texPinName.c_str() ,"mz.fb.Texture", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_ONLY, pinCategory.c_str(),0,&texData),
			fb::CreatePinDirect(fbb, &opacityId, opacityPinName.c_str(), "float", fb::ShowAs::PROPERTY, fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY, pinCategory.c_str(),0,&opacityData),
			fb::CreatePinDirect(fbb, &blendId, blendPinName.c_str(), "mz.fb.BlendMode", fb::ShowAs::PROPERTY, fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY, pinCategory.c_str(),0,&blendModeData),
		};
		mzEngine.HandleEvent(CreateAppEvent(fbb,CreatePartialNodeUpdateDirect(fbb, &((MergeContext*)(ctx))->Id, ClearFlags::NONE,0,&pins)));
	}
};

}

void RegisterMerge(MzNodeFunctions* out)
{
	out->TypeName = "mz.utilities.Merge";
	out->OnMenuRequested = [](void* ctx, const MzContextMenuRequest* request) {
		((mz::filters::MergeContext*)ctx)->OnMenuRequested(ctx, request);
	};
	out->OnMenuCommand = [](void* ctx, uint32_t cmd) {
		((mz::filters::MergeContext*)ctx)->OnMenuCommand(ctx, cmd);
	};
	out->ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args) {
		((mz::filters::MergeContext*)ctx)->Run(args);
		return MZ_RESULT_SUCCESS;
	};
}
