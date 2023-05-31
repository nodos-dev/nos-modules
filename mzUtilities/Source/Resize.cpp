#include <MediaZ/Helpers.hpp>

#include "Resize.frag.spv.dat"

#include "Builtins_generated.h"

namespace mz::utilities
{

struct ResizeContext
{
	MzUUID NodeId;
	
	static void OnNodeUpdated(void* ctx, const MzFbNode* updatedNode)
	{
		mzEngine.LogW("UPDATED RESIZE NODE");
		updatedNode->UnPackTo((fb::TNode*)ctx);
	}

	static void OnNodeDeleted(void* ctx, MzUUID nodeId)
	{
		mzEngine.LogW("DELETED RESIZE NODE");
		delete (fb::TNode*)ctx;
	}

	static MzResult GetPasses(size_t* outCount, MzPassInfo* infos)
	{
		*outCount = 1;
		if(!infos)
			return MZ_RESULT_SUCCESS;

		infos->Key = "Resize_Pass";
		infos->Shader = "Resize_Shader";
		infos->Blend = false;
		infos->MultiSample = 1;

		return MZ_RESULT_SUCCESS;
	}

	static MzResult GetShaders(size_t* outCount, const char** outShaderNames, MzBuffer* outSpirvBufs)
	{
		*outCount = 1;
		if(!outSpirvBufs || !outShaderNames)
			return MZ_RESULT_SUCCESS;
		
		*outShaderNames = "Resize_Shader";
		outSpirvBufs->Data = (void*)(Resize_frag_spv);
		outSpirvBufs->Size = sizeof(Resize_frag_spv);
		return MZ_RESULT_SUCCESS;
	}
	
	static MzResult ExecuteNode(void* ctx, const MzNodeExecuteArgs* args)
	{
		mzEngine.LogW("EXECUTE RESIZE NODE");
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
			auto texFb = ConvertTextureInfo(tex);
			texFb.unscaled = true;
			auto texFbBuf = mz::Buffer::From(texFb);
			mzEngine.SetPinValue(((MzUUID)(args->PinIds[1])), {.Data = texFbBuf.data(), .Size = texFbBuf.size()});
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

		return MZ_RESULT_SUCCESS;
	}
	
};

void RegisterResize(MzNodeFunctions* out)
{
	out->TypeName = "mz.utilities.Resize";
	out->GetPasses = mz::utilities::ResizeContext::GetPasses;
	out->GetShaders = mz::utilities::ResizeContext::GetShaders;
	out->ExecuteNode = [](void* ctx, const MzNodeExecuteArgs* args)-> MzResult {
		return ((mz::utilities::ResizeContext*)ctx)->ExecuteNode(ctx, args);
	};
	out->OnNodeDeleted = mz::utilities::ResizeContext::OnNodeDeleted;
	out->OnNodeUpdated = mz::utilities::ResizeContext::OnNodeUpdated;
}

} // namespace mz::utilities


