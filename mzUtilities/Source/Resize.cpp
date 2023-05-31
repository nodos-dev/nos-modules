#include "Resize.hpp"
#include "Resize.frag.spv.dat"

#include "Builtins_generated.h"

namespace mz::utilities
{

struct ResizeContext
{
	static void OnNodeUpdated(void* ctx, const MzFbNode* updatedNode)
	{
		updatedNode->UnPackTo((fb::TNode*)ctx);
	}

	static void OnNodeDeleted(void* ctx, MzUUID nodeId)
	{
		delete (fb::TNode*)ctx;
	}

	static void OnNodeCreated(const MzFbNode* node, void** outCtxPtr)
	{
		static bool reg = false;
		if(reg) return;
		reg = true;
	}

	static MzResult GetShaders(size_t* outCount, const char** outShaderNames, MzBuffer* outSpirvBufs)
	{
		*outCount = 1;
		*outShaderNames = "Resize_Pass";
		outSpirvBufs->Data = (void*)(Resize_frag_spv);
		outSpirvBufs->Size = sizeof(Resize_frag_spv);
		return MZ_RESULT_SUCCESS;
	}
	
	static void ExecuteNode(void* ctx, const MzNodeExecuteArgs* args)
	{
		auto pins = GetPinValues(args);

		auto inputTex = DeserializeTextureInfo(pins["Input"]);
		auto method = GetPinValue<uint32_t>(pins, "Method");
		
		auto tex = DeserializeTextureInfo(pins["Output"]);
		auto size = GetPinValue<mz::fb::vec2>(pins, "Size");
		
		if(size->x() != tex.info.texture.width ||
			size->y() != tex.info.texture.height)
		{
			tex.info.texture.width = size->x();
			tex.info.texture.height = size->y();
			mzEngine.Destroy(&tex);
			mzEngine.Create(&tex);
			
			MzBuffer buf = {
				.Data = &tex,
				.Size = sizeof(tex)
			};
			mzEngine.SetPinValue(((MzUUID)(args->PinIds[1])), buf);
		}

		std::vector bindings = {
			ShaderBinding("Input", inputTex),
			ShaderBinding("Method", method)
		};
		
		MzRunPassParams resizeParam {
			.PassKey = "Resize_Pass",
			.Bindings = bindings.data(),
			.BindingCount = 2,
			.Output = tex,
			.Wireframe = 0,
			.Benchmark = 0,
		};

		mzEngine.RunPass(nullptr, &resizeParam);
	}
	
};

} // namespace mz::utilities

void RegisterResize(MzNodeFunctions* out)
{
	
}
