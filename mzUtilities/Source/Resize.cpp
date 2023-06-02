#include <MediaZ/Helpers.hpp>

#include "Resize.frag.spv.dat"

#include "Builtins_generated.h"

namespace mz::utilities
{

MZ_REGISTER_NAME2(Resize_Pass);
MZ_REGISTER_NAME2(Resize_Shader);
MZ_REGISTER_NAME2(Input);
MZ_REGISTER_NAME2(Method);

struct ResizeContext
{
	mzUUID NodeId;
	
	static void OnNodeUpdated(void* ctx, const mzFbNode* updatedNode)
	{
		mzEngine.LogW("UPDATED RESIZE NODE");
		updatedNode->UnPackTo((fb::TNode*)ctx);
	}

	static void OnNodeDeleted(void* ctx, mzUUID nodeId)
	{
		mzEngine.LogW("DELETED RESIZE NODE");
		delete (fb::TNode*)ctx;
	}

	static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
	{
		*outCount = 1;
		if(!infos)
			return MZ_RESULT_SUCCESS;

		infos->Key    = Resize_Pass_Name;
		infos->Shader = Resize_Shader_Name;
		infos->Blend = false;
		infos->MultiSample = 1;

		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetShaders(size_t* outCount, mzName* outShaderNames, mzBuffer* outSpirvBufs)
	{
		*outCount = 1;
		if(!outSpirvBufs || !outShaderNames)
			return MZ_RESULT_SUCCESS;
		
		*outShaderNames = Resize_Shader_Name;
		outSpirvBufs->Data = (void*)(Resize_frag_spv);
		outSpirvBufs->Size = sizeof(Resize_frag_spv);
		return MZ_RESULT_SUCCESS;
	}
	
	static mzResult ExecuteNode(void* ctx, const mzNodeExecuteArgs* args)
	{
		mzEngine.LogW("EXECUTE RESIZE NODE");
		auto pins = GetPinValues(args);
		MZ_REGISTER_NAME2(Input);
		MZ_REGISTER_NAME2(Method);
		MZ_REGISTER_NAME2(Output);
		MZ_REGISTER_NAME2(Size);
		auto inputTex = DeserializeTextureInfo(pins[Input_Name]);
		auto method = GetPinValue<uint32_t>(pins, Method_Name);
		
		auto tex = DeserializeTextureInfo(pins[Output_Name]);
		auto size = GetPinValue<mzVec2u>(pins, Size_Name);
		
		if(size->x != tex.Info.Texture.Width ||
			size->y != tex.Info.Texture.Height)
		{
			tex.Info.Texture.Width = size->x;
			tex.Info.Texture.Height = size->y;
			mzEngine.Destroy(&tex);
			mzEngine.Create(&tex);
			auto texFb = ConvertTextureInfo(tex);
			texFb.unscaled = true;
			auto texFbBuf = mz::Buffer::From(texFb);
			mzEngine.SetPinValue(((mzUUID)(args->PinIds[1])), {.Data = texFbBuf.data(), .Size = texFbBuf.size()});
		}

		std::vector bindings = {
			ShaderBinding(Input_Name, inputTex),
			ShaderBinding(Method_Name, method)
		};
		
		mzRunPassParams resizeParam {
			.Key = Resize_Pass_Name,
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

void RegisterResize(mzNodeFunctions* out)
{
	out->TypeName = "mz.utilities.Resize";
	out->GetPasses = mz::utilities::ResizeContext::GetPasses;
	out->GetShaders = mz::utilities::ResizeContext::GetShaders;
	out->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args)-> mzResult {
		return ((mz::utilities::ResizeContext*)ctx)->ExecuteNode(ctx, args);
	};
	out->OnNodeDeleted = mz::utilities::ResizeContext::OnNodeDeleted;
	out->OnNodeUpdated = mz::utilities::ResizeContext::OnNodeUpdated;
}

} // namespace mz::utilities


