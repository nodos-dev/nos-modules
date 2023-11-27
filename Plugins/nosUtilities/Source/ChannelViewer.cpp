// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/Helpers.hpp>
#include "../Shaders/ChannelViewer.frag.spv.dat"
#include <glm/glm.hpp>


NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(Channel);
NOS_REGISTER_NAME(Format);
NOS_REGISTER_NAME(Channel_Viewer_Pass);
NOS_REGISTER_NAME(Channel_Viewer_Shader);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_ChannelViewer, "nos.utilities.ChannelViewer")
namespace nos::utilities
{

static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
{
	*outCount = 1;
	if (!outShaders)
		return NOS_RESULT_SUCCESS;

	outShaders[0] = {.Key = NSN_Channel_Viewer_Shader, .Source = {.SpirvBlob = {(void*)ChannelViewer_frag_spv, sizeof(ChannelViewer_frag_spv)}}};
	return NOS_RESULT_SUCCESS;
}

static nosResult GetPasses(size_t* outCount, nosPassInfo* infos)
{
	*outCount = 1;
	if (!infos)
		return NOS_RESULT_SUCCESS;

	infos->Key = NSN_Channel_Viewer_Pass;
	infos->Shader = NSN_Channel_Viewer_Shader;
	infos->Blend = false;
	infos->MultiSample = 1;

	return NOS_RESULT_SUCCESS;
}

static nosResult ExecuteNode(void* ctx, const nosNodeExecuteArgs* pins)
{
	auto values = GetPinValues(pins);
	const nosResourceShareInfo input = DeserializeTextureInfo(values[NSN_Input]);
	const nosResourceShareInfo output = DeserializeTextureInfo(values[NSN_Output]);

	auto channel = *(u32*)values[NSN_Channel];
	auto format = *(u32*)values[NSN_Format];

	glm::vec4 val{};
	val[channel & 3] = 1;

	constexpr glm::vec3 coeffs[3] = {{.299f, .587f, .114f}, {.2126f, .7152f, .0722f}, {.2627f, .678f, .0593f}};

	glm::vec4 multipliers = glm::vec4(coeffs[format], channel > 3);
	std::vector<nosShaderBinding> bindings = {
											ShaderBinding(NSN_Input, input),
											 ShaderBinding(NSN_Channel, val), 
											 ShaderBinding(NSN_Format, multipliers)};

	nosRunPassParams pass = {
		.Key = NSN_Channel_Viewer_Pass,
		.Bindings = bindings.data(),
		.BindingCount = (u32)bindings.size(),
		.Output = output,
		.Wireframe = false,
	};
	nosEngine.RunPass(0, &pass);
	return NOS_RESULT_SUCCESS;
}

void RegisterChannelViewer(nosNodeFunctions* out)
{
	out->TypeName = NSN_Nos_Utilities_ChannelViewer;
	out->GetShaders = nos::utilities::GetShaders;
	out->GetPasses = nos::utilities::GetPasses;
	out->ExecuteNode = nos::utilities::ExecuteNode;
}


} 

