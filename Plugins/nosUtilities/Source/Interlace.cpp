// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/Helpers.hpp>

#include "Interlace.frag.spv.dat"
#include "Deinterlace.frag.spv.dat"

NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
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

	InterlaceNode(nosFbNode const* node) : NodeContext(node) {}

	~InterlaceNode() {}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) override {}

	nosResult BeginCopyFrom(nosCopyInfo* copyInfo) override {
		copyInfo->CopyTextureFrom.Info.Texture.FieldType = Field;
		Field = FlippedField(Field);
		return NOS_RESULT_SUCCESS;
	}

	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args) {
		auto pinIds = GetPinIds(args);
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = DeserializeTextureInfo(pinValues[NSN_Input]);
		auto outputTextureInfo = DeserializeTextureInfo(pinValues[NSN_Output]);
		nosRunPassParams interlacePass = {};
		interlacePass.Key = NSN_Utilities_Interlace_Pass;
		uint32_t isOdd = Field - 1;
		std::vector bindings = {
			ShaderBinding(NSN_Input, inputTextureInfo),
			ShaderBinding(NSN_ShouldOutputOdd, isOdd),
		};
		interlacePass.Bindings = bindings.data();
		interlacePass.BindingCount = bindings.size();
		interlacePass.Output = outputTextureInfo;
		nosEngine.RunPass(0, &interlacePass);
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return NOS_RESULT_SUCCESS;
		outShaders[0] = {.Key = NSN_Utilities_Interlace_Fragment_Shader,
						 .Source = {.SpirvBlob = {(void*)Interlace_frag_spv, sizeof(Interlace_frag_spv)}}};
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetPasses(size_t* count, nosPassInfo* passes)
	{
		*count = 1;
		if (!passes)
			return NOS_RESULT_SUCCESS;
		*passes = nosPassInfo{
			.Key = NSN_Utilities_Interlace_Pass,
			.Shader = NSN_Utilities_Interlace_Fragment_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};
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
	DeinterlaceNode(nosFbNode const* node) : NodeContext(node) {}

	~DeinterlaceNode() {}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) override {}

	nosResult BeginCopyFrom(nosCopyInfo* copyInfo) override {
		copyInfo->CopyTextureFrom.Info.Texture.FieldType = NOS_TEXTURE_FIELD_TYPE_PROGRESSIVE;
		return NOS_RESULT_SUCCESS;
	}

	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args) {
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = DeserializeTextureInfo(pinValues[NSN_Input]);
		auto outputTextureInfo = DeserializeTextureInfo(pinValues[NSN_Output]);
		nosRunPassParams deinterlacePass = {};
		deinterlacePass.Key = NSN_Utilities_Deinterlace_Pass;
		auto field = inputTextureInfo.Info.Texture.FieldType;
		bool isInterlaced = IsTextureFieldTypeInterlaced(field);
		if (!isInterlaced) {
			nosEngine.LogW("Deinterlace Node: Input is not interlaced!");
			return NOS_RESULT_FAILED;
		}
		uint32_t isOdd = field - 1;
		std::vector bindings = {
			ShaderBinding(NSN_Input, inputTextureInfo),
			ShaderBinding(NSN_IsOdd, isOdd)
		};
		deinterlacePass.Bindings = bindings.data();
		deinterlacePass.BindingCount = bindings.size();
		deinterlacePass.Output = outputTextureInfo;
		deinterlacePass.DoNotClear = true;
		nosEngine.RunPass(0, &deinterlacePass);
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return NOS_RESULT_SUCCESS;
		outShaders[0] = { .Key = NSN_Utilities_Deinterlace_Fragment_Shader,
						 .Source = {.SpirvBlob = {(void*)Deinterlace_frag_spv, sizeof(Deinterlace_frag_spv)}} };
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetPasses(size_t* count, nosPassInfo* passes)
	{
		*count = 1;
		if (!passes)
			return NOS_RESULT_SUCCESS;
		*passes = nosPassInfo {
			.Key = NSN_Utilities_Deinterlace_Pass,
			.Shader = NSN_Utilities_Deinterlace_Fragment_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};
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

void RegisterInterlace(nosNodeFunctions* nodeFunctions)
{
	NOS_BIND_NODE_CLASS(NSN_TypeName_Utilities_Interlace, InterlaceNode, nodeFunctions);
}

void RegisterDeinterlace(nosNodeFunctions* nodeFunctions)
{
	NOS_BIND_NODE_CLASS(NSN_TypeName_Utilities_Deinterlace, DeinterlaceNode, nodeFunctions);
}

} // namespace nos::utilities
