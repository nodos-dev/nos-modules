// Copyright MediaZ AS. All Rights Reserved.
#include <MediaZ/Helpers.hpp>
#include "ChannelViewer.frag.spv.dat"
#include <glm/glm.hpp>


MZ_REGISTER_NAME2(Input);
MZ_REGISTER_NAME2(Output);
MZ_REGISTER_NAME2(Channel);
MZ_REGISTER_NAME2(Format);
MZ_REGISTER_NAME2(Channel_Viewer_Pass);
MZ_REGISTER_NAME2(Channel_Viewer_Shader);

namespace mz::utilities
{

static mzResult GetShaders(size_t* outCount, mzName* outShaderNames, mzBuffer* infos)
{
	*outCount = 1;
	if (!infos)
		return MZ_RESULT_SUCCESS;

	outShaderNames[0] = Channel_Viewer_Shader_Name;
	infos->Data = (void*)ChannelViewer_frag_spv;
	infos->Size = sizeof(ChannelViewer_frag_spv);

	return MZ_RESULT_SUCCESS;
}

static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
{
	*outCount = 1;
	if (!infos)
		return MZ_RESULT_SUCCESS;

	infos->Key    = Channel_Viewer_Pass_Name;
	infos->Shader = Channel_Viewer_Shader_Name;
	infos->Blend = false;
	infos->MultiSample = 1;

	return MZ_RESULT_SUCCESS;
}

static mzResult Run(void* ctx, const mzNodeExecuteArgs* pins)
{
	auto values = GetPinValues(pins);
	const mzResourceShareInfo input = DeserializeTextureInfo(values[Input_Name]);
	const mzResourceShareInfo output = DeserializeTextureInfo(values[Output_Name]);

	auto channel = *(u32*)values[Channel_Name];
	auto format = *(u32*)values[Format_Name];

	glm::vec4 val{};
	val[channel & 3] = 1;

	constexpr glm::vec3 coeffs[3] = {{.299f, .587f, .114f}, {.2126f, .7152f, .0722f}, {.2627f, .678f, .0593f}};

	glm::vec4 multipliers = glm::vec4(coeffs[format], channel > 3);
	std::vector<mzShaderBinding> bindings = {
		ShaderBinding(Input_Name, input), ShaderBinding(Channel_Name, val), ShaderBinding(Format_Name, multipliers)};

	mzRunPassParams pass = {
		.Key = Channel_Viewer_Pass_Name,
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
	out->TypeName = MZ_NAME_STATIC("mz.utilities.ChannelViewer");
	out->GetShaders = mz::utilities::GetShaders;
	out->GetPasses = mz::utilities::GetPasses;
	out->ExecuteNode = mz::utilities::Run;
}


} 

