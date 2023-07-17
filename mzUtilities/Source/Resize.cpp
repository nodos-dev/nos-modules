#include <MediaZ/Helpers.hpp>

#include "Resize.frag.spv.dat"

#include "Builtins_generated.h"

namespace mz::utilities
{

MZ_REGISTER_NAME(Resize_Pass);
MZ_REGISTER_NAME(Resize_Shader);
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Method);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(Size);
MZ_REGISTER_NAME_SPACED(Mz_Utilities_Resize, "mz.utilities.Resize")

static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
{
	*outCount = 1;
	if(!infos)
		return MZ_RESULT_SUCCESS;

	infos->Key = MZN_Resize_Pass;
	infos->Shader = MZN_Resize_Shader;
	infos->Blend = false;
	infos->MultiSample = 1;

	return MZ_RESULT_SUCCESS;
}

static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
{
    *outCount = 1;
    if (!outShaders)
        return MZ_RESULT_SUCCESS;

    outShaders[0] = {.Key=MZN_Resize_Shader, .SpirvBlob = {(void*)Resize_frag_spv, sizeof(Resize_frag_spv)}};
    return MZ_RESULT_SUCCESS;
}

static mzResult ExecuteNode(void* ctx, const mzNodeExecuteArgs* args)
{
	auto pins = GetPinValues(args);
	auto inputTex = DeserializeTextureInfo(pins[MZN_Input]);
	auto method = GetPinValue<uint32_t>(pins, MZN_Method);
		
	auto tex = DeserializeTextureInfo(pins[MZN_Output]);
	auto size = GetPinValue<mzVec2u>(pins, MZN_Size);
		
	if(size->x != tex.Info.Texture.Width ||
		size->y != tex.Info.Texture.Height)
	{
		auto prevTex = tex;
		tex.Memory = {};
		tex.Info.Texture.Width = size->x;
		tex.Info.Texture.Height = size->y;
		mzResult re;
		re = mzEngine.Create(&tex);
		assert(MZ_RESULT_SUCCESS == re);
		re = mzEngine.Destroy(&prevTex);
		assert(MZ_RESULT_SUCCESS == re);

		auto texFb = ConvertTextureInfo(tex);
		texFb.unscaled = true;
		auto texFbBuf = mz::Buffer::From(texFb);
		mzEngine.SetPinValue(args->PinIds[1], {.Data = texFbBuf.data(), .Size = texFbBuf.size()});
	}

	std::vector bindings = {ShaderBinding(MZN_Input, inputTex), ShaderBinding(MZN_Method, method)};
		
	mzRunPassParams resizeParam {
		.Key = MZN_Resize_Pass,
		.Bindings = bindings.data(),
		.BindingCount = 2,
		.Output = tex,
		.Wireframe = 0,
		.Benchmark = 0,
	};

	mzEngine.RunPass(nullptr, &resizeParam);

	return MZ_RESULT_SUCCESS;
}

void RegisterResize(mzNodeFunctions* out)
{
	out->TypeName = MZN_Mz_Utilities_Resize;
	out->GetPasses = GetPasses;
	out->GetShaders = GetShaders;
	out->ExecuteNode = ExecuteNode;
}

} // namespace mz::utilities


