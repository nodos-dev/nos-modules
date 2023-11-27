#include <Nodos/Helpers.hpp>

#include "../Shaders/Resize.frag.spv.dat"

#include "Builtins_generated.h"

namespace nos::utilities
{

NOS_REGISTER_NAME(Resize_Pass);
NOS_REGISTER_NAME(Resize_Shader);
NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Method);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(Size);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_Resize, "nos.utilities.Resize")

static nosResult GetPasses(size_t* outCount, nosPassInfo* infos)
{
	*outCount = 1;
	if(!infos)
		return NOS_RESULT_SUCCESS;

	infos->Key = NSN_Resize_Pass;
	infos->Shader = NSN_Resize_Shader;
	infos->Blend = false;
	infos->MultiSample = 1;

	return NOS_RESULT_SUCCESS;
}

static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
{
    *outCount = 1;
    if (!outShaders)
        return NOS_RESULT_SUCCESS;

	outShaders[0] = {.Key=NSN_Resize_Shader, .Source = {.SpirvBlob = {(void*)Resize_frag_spv, sizeof(Resize_frag_spv)}}};
    return NOS_RESULT_SUCCESS;
}

static nosResult ExecuteNode(void* ctx, const nosNodeExecuteArgs* args)
{
	auto pins = GetPinValues(args);
	auto inputTex = DeserializeTextureInfo(pins[NSN_Input]);
	auto method = GetPinValue<uint32_t>(pins, NSN_Method);
		
	auto tex = DeserializeTextureInfo(pins[NSN_Output]);
	auto size = GetPinValue<nosVec2u>(pins, NSN_Size);
		
	if(size->x != tex.Info.Texture.Width ||
		size->y != tex.Info.Texture.Height)
	{
		auto prevTex = tex;
		prevTex.Memory = {};
		prevTex.Info.Texture.Width = size->x;
		prevTex.Info.Texture.Height = size->y;
		auto texFb = ConvertTextureInfo(prevTex);
		texFb.unscaled = true;
		auto texFbBuf = nos::Buffer::From(texFb);
		nosEngine.SetPinValue(args->PinIds[1], {.Data = texFbBuf.data(), .Size = texFbBuf.size()});
	}

	std::vector bindings = {ShaderBinding(NSN_Input, inputTex), ShaderBinding(NSN_Method, method)};
		
	nosRunPassParams resizeParam {
		.Key = NSN_Resize_Pass,
		.Bindings = bindings.data(),
		.BindingCount = 2,
		.Output = tex,
		.Wireframe = 0,
		.Benchmark = 0,
	};

	nosEngine.RunPass(nullptr, &resizeParam);

	return NOS_RESULT_SUCCESS;
}

void RegisterResize(nosNodeFunctions* out)
{
	out->TypeName = NSN_Nos_Utilities_Resize;
	out->GetPasses = GetPasses;
	out->GetShaders = GetShaders;
	out->ExecuteNode = ExecuteNode;
}

} // namespace nos::utilities


