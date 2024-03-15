// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/Helpers.hpp>
#include "Interlace.frag.spv.dat"
#include "Deinterlace.frag.spv.dat"

#include "Names.h"

NOS_REGISTER_NAME(ShouldOutputOdd);
NOS_REGISTER_NAME(IsOdd);
NOS_REGISTER_NAME_SPACED(TypeName_Utilities_Interlace, "nos.utilities.Interlace")
NOS_REGISTER_NAME_SPACED(TypeName_Utilities_Deinterlace, "nos.utilities.Deinterlace")

NOS_REGISTER_NAME(Utilities_Interlace_Fragment_Shader);
NOS_REGISTER_NAME(Utilities_Interlace_Pass);

NOS_REGISTER_NAME(Utilities_Deinterlace_Fragment_Shader);
NOS_REGISTER_NAME(Utilities_Deinterlace_Pass);

namespace nos::utilities
{
struct InterlaceNode : NodeContext
{
	nosTextureFieldType Field = NOS_TEXTURE_FIELD_TYPE_EVEN;

	InterlaceNode(nosFbNode const* node)
		: NodeContext(node)
	{
	}

	~InterlaceNode()
	{
	}

	nosResult CopyFrom(nosCopyInfo* copyInfo) override
	{
		vkss::SetFieldType(copyInfo->ID, *copyInfo->PinData, Field);
		Field = vkss::FlippedField(Field);
		return NOS_RESULT_SUCCESS;
	}

	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args)
	{
		auto pinIds = GetPinIds(args);
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = vkss::DeserializeTextureInfo(pinValues[NSN_Input]);
		auto outputTextureInfo = vkss::DeserializeTextureInfo(pinValues[NSN_Output]);
		nosRunPassParams interlacePass = {};
		interlacePass.Key = NSN_Utilities_Interlace_Pass;
		uint32_t isOdd = Field - 1;
		std::vector bindings = {
			vkss::ShaderBinding(NSN_Input, inputTextureInfo),
			vkss::ShaderBinding(NSN_ShouldOutputOdd, isOdd),
		};
		interlacePass.Bindings = bindings.data();
		interlacePass.BindingCount = bindings.size();
		interlacePass.Output = outputTextureInfo;
		nosVulkan->RunPass(0, &interlacePass);
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
	{
		*count = 0;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;
		return NOS_RESULT_SUCCESS;
	}
};

struct DeinterlaceNode : NodeContext
{
	DeinterlaceNode(nosFbNode const* node)
		: NodeContext(node)
	{
	}

	~DeinterlaceNode()
	{
	}

	nosResult CopyFrom(nosCopyInfo* copyInfo) override
	{
		vkss::SetFieldType(copyInfo->ID, *copyInfo->PinData, NOS_TEXTURE_FIELD_TYPE_PROGRESSIVE);
		return NOS_RESULT_SUCCESS;
	}

	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args)
	{
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = vkss::DeserializeTextureInfo(pinValues[NSN_Input]);
		auto outputTextureInfo = vkss::DeserializeTextureInfo(pinValues[NSN_Output]);
		nosRunPassParams deinterlacePass = {};
		deinterlacePass.Key = NSN_Utilities_Deinterlace_Pass;
		auto field = inputTextureInfo.Info.Texture.FieldType;
		bool isInterlaced = vkss::IsTextureFieldTypeInterlaced(field);
		if (!isInterlaced)
		{
			nosEngine.LogW("Deinterlace Node: Input is not interlaced!");
			return NOS_RESULT_FAILED;
		}
		uint32_t isOdd = field - 1;
		std::vector bindings = {
			vkss::ShaderBinding(NSN_Input, inputTextureInfo),
			vkss::ShaderBinding(NSN_IsOdd, isOdd)
		};
		deinterlacePass.Bindings = bindings.data();
		deinterlacePass.BindingCount = bindings.size();
		deinterlacePass.Output = outputTextureInfo;
		deinterlacePass.DoNotClear = true;
		nosVulkan->RunPass(0, &deinterlacePass);
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
	{
		*count = 0;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterInterlace(nosNodeFunctions* nodeFunctions)
{
	NOS_BIND_NODE_CLASS(NSN_TypeName_Utilities_Interlace, InterlaceNode, nodeFunctions);
	nosShaderInfo shader = {.Key = NSN_Utilities_Interlace_Fragment_Shader,
	                        .Source = {.SpirvBlob = {(void*)Interlace_frag_spv, sizeof(Interlace_frag_spv)}}};
	auto ret = nosVulkan->RegisterShaders(1, &shader);
	if (NOS_RESULT_SUCCESS != ret)
		return ret;
	nosPassInfo pass = {.Key = NSN_Utilities_Interlace_Pass,
	                    .Shader = NSN_Utilities_Interlace_Fragment_Shader,
	                    .MultiSample = 1,};
	return nosVulkan->RegisterPasses(1, &pass);
}

nosResult RegisterDeinterlace(nosNodeFunctions* nodeFunctions)
{
	NOS_BIND_NODE_CLASS(NSN_TypeName_Utilities_Deinterlace, DeinterlaceNode, nodeFunctions);

	nosShaderInfo shader = {.Key = NSN_Utilities_Deinterlace_Fragment_Shader,
	                        .Source = {.SpirvBlob = {(void*)Deinterlace_frag_spv, sizeof(Deinterlace_frag_spv)}}};
	auto ret = nosVulkan->RegisterShaders(1, &shader);
	if (NOS_RESULT_SUCCESS != ret)
		return ret;
	nosPassInfo pass = {
		.Key = NSN_Utilities_Deinterlace_Pass,
		.Shader = NSN_Utilities_Deinterlace_Fragment_Shader,
		.MultiSample = 1,};
	return nosVulkan->RegisterPasses(1, &pass);
}

} // namespace nos::utilities
