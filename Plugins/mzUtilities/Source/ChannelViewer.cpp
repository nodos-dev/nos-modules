// Copyright MediaZ AS. All Rights Reserved.
#include <MediaZ/Helpers.hpp>
#include "../Shaders/ChannelViewer.frag.spv.dat"
#include <glm/glm.hpp>


MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(Channel);
MZ_REGISTER_NAME(Format);
MZ_REGISTER_NAME(Channel_Viewer_Pass);
MZ_REGISTER_NAME(Channel_Viewer_Shader);
MZ_REGISTER_NAME_SPACED(Mz_Utilities_ChannelViewer, "mz.utilities.ChannelViewer")
namespace mz::utilities
{

static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
{
	*outCount = 1;
	if (!outShaders)
		return MZ_RESULT_SUCCESS;

	outShaders[0] = {.Key = MZN_Channel_Viewer_Shader, .Source = {.SpirvBlob = {(void*)ChannelViewer_frag_spv, sizeof(ChannelViewer_frag_spv)}}};
	return MZ_RESULT_SUCCESS;
}

static mzResult GetPasses(size_t* outCount, mzPassInfo* infos)
{
	*outCount = 1;
	if (!infos)
		return MZ_RESULT_SUCCESS;

	infos->Key = MZN_Channel_Viewer_Pass;
	infos->Shader = MZN_Channel_Viewer_Shader;
	infos->Blend = false;
	infos->MultiSample = 1;

	return MZ_RESULT_SUCCESS;
}

static mzResult ExecuteNode(void* ctx, const mzNodeExecuteArgs* pins)
{
	auto values = GetPinValues(pins);
	const mzResourceShareInfo input = DeserializeTextureInfo(values[MZN_Input]);
	const mzResourceShareInfo output = DeserializeTextureInfo(values[MZN_Output]);

	auto channel = *(u32*)values[MZN_Channel];
	auto format = *(u32*)values[MZN_Format];

	glm::vec4 val{};
	val[channel & 3] = 1;

	constexpr glm::vec3 coeffs[3] = {{.299f, .587f, .114f}, {.2126f, .7152f, .0722f}, {.2627f, .678f, .0593f}};

	glm::vec4 multipliers = glm::vec4(coeffs[format], channel > 3);
	std::vector<mzShaderBinding> bindings = {
											ShaderBinding(MZN_Input, input),
											 ShaderBinding(MZN_Channel, val), 
											 ShaderBinding(MZN_Format, multipliers)};

	mzRunPassParams pass = {
		.Key = MZN_Channel_Viewer_Pass,
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
	out->TypeName = MZN_Mz_Utilities_ChannelViewer;
	out->GetShaders = mz::utilities::GetShaders;
	out->GetPasses = mz::utilities::GetPasses;
	out->ExecuteNode = mz::utilities::ExecuteNode;
}


} 

