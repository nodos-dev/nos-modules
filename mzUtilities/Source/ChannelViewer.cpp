// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>
#include "ChannelViewer.frag.spv.dat"
#include <glm/glm.hpp>

namespace mz::utilities
{

static mzResult GetShaders(size_t* outCount, const char** outShaderNames, mzBuffer* infos)
{
	*outCount = 1;
	if (!infos)
		return MZ_RESULT_SUCCESS;

	outShaderNames[0] = "Channel_Viewer";
	infos->Data = (void*)ChannelViewer_frag_spv;
	infos->Size = sizeof(ChannelViewer_frag_spv);

	return MZ_RESULT_SUCCESS;
}

static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
{
	*outCount = 1;
	if (!infos)
		return MZ_RESULT_SUCCESS;

	infos->Key = "Channel_Viewer_Pass";
	infos->Shader = "Channel_Viewer";
	infos->Blend = false;
	infos->MultiSample = 1;

	return MZ_RESULT_SUCCESS;
}

static mzResult Run(void* ctx, const mzNodeExecuteArgs* pins)
{
	auto values = GetPinValues(pins);
	const mzResourceShareInfo input = DeserializeTextureInfo(values["Input"]);
	const mzResourceShareInfo output = DeserializeTextureInfo(values["Output"]);

	auto channel = *(u32*)values["Channel"];
	auto format = *(u32*)values["Format"];

	glm::vec4 val{};
	val[channel & 3] = 1;

	constexpr glm::vec3 coeffs[3] = {{.299f, .587f, .114f}, {.2126f, .7152f, .0722f}, {.2627f, .678f, .0593f}};

	glm::vec4 multipliers = glm::vec4(coeffs[format], channel > 3);
	std::vector<mzShaderBinding> bindings = {
		ShaderBinding("Input", input), ShaderBinding("Channel", val), ShaderBinding("Format", multipliers)};

	mzRunPassParams pass = {
		.PassKey = "Channel_Viewer_Pass",
		.Bindings = bindings.data(),
		.BindingCount = (u32)bindings.size(),
		.Output = output,
		.Wireframe = false,
	};
	mzEngine.RunPass(0, &pass);
	return MZ_RESULT_SUCCESS;
}

void RegisterChannelViewer(mzNodeFunctions* out)
{
	out->TypeName = "mz.utilities.ChannelViewer";
	out->GetShaders = mz::utilities::GetShaders;
	out->GetPasses = mz::utilities::GetPasses;
	out->ExecuteNode = mz::utilities::Run;
}


} 

