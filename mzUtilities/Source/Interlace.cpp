// Copyright MediaZ AS. All Rights Reserved.
#include <MediaZ/Helpers.hpp>

#include "Interlace.frag.spv.dat"
#include "Deinterlace.frag.spv.dat"

MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(Output);
MZ_REGISTER_NAME(ShouldOutputOdd);
MZ_REGISTER_NAME(IsOdd);
MZ_REGISTER_NAME_SPACED(TypeName_Utilities_Interlace, "mz.utilities.Interlace")
MZ_REGISTER_NAME_SPACED(TypeName_Utilities_Deinterlace, "mz.utilities.Deinterlace")

MZ_REGISTER_NAME(Utilities_Interlace_Fragment_Shader);
MZ_REGISTER_NAME(Utilities_Interlace_Pass);

MZ_REGISTER_NAME(Utilities_Deinterlace_Fragment_Shader);
MZ_REGISTER_NAME(Utilities_Deinterlace_Pass);

namespace mz::utilities
{

struct InterlaceNode : NodeContext
{
	mzTextureFieldType Field = MZ_TEXTURE_FIELD_TYPE_EVEN;

	InterlaceNode(mzFbNode const* node) : NodeContext(node) {}

	~InterlaceNode() {}

	void OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {}

	mzResult BeginCopyFrom(mzCopyInfo* copyInfo) override {
		copyInfo->CopyTextureFrom.Info.Texture.FieldType = Field;
		Field = FlippedField(Field);
		return MZ_RESULT_SUCCESS;
	}

	virtual mzResult ExecuteNode(const mzNodeExecuteArgs* args) {
		auto pinIds = GetPinIds(args);
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = DeserializeTextureInfo(pinValues[MZN_Input]);
		auto outputTextureInfo = DeserializeTextureInfo(pinValues[MZN_Output]);
		mzRunPassParams interlacePass = {};
		interlacePass.Key = MZN_Utilities_Interlace_Pass;
		uint32_t isOdd = Field - 1;
		std::vector bindings = {
			ShaderBinding(MZN_Input, inputTextureInfo),
			ShaderBinding(MZN_ShouldOutputOdd, isOdd),
		};
		interlacePass.Bindings = bindings.data();
		interlacePass.BindingCount = bindings.size();
		interlacePass.Output = outputTextureInfo;
		mzEngine.RunPass(0, &interlacePass);
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;
		outShaders[0] = {.Key = MZN_Utilities_Interlace_Fragment_Shader,
						 .Source = {.SpirvBlob = {(void*)Interlace_frag_spv, sizeof(Interlace_frag_spv)}}};
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes)
	{
		*count = 1;
		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes = mzPassInfo{
			.Key = MZN_Utilities_Interlace_Pass,
			.Shader = MZN_Utilities_Interlace_Fragment_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns)
	{
		*count = 0;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;
		return MZ_RESULT_SUCCESS;
	}
};

struct DeinterlaceNode : NodeContext
{
	DeinterlaceNode(mzFbNode const* node) : NodeContext(node) {}

	~DeinterlaceNode() {}

	void OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {}

	mzResult BeginCopyFrom(mzCopyInfo* copyInfo) override {
		copyInfo->CopyTextureFrom.Info.Texture.FieldType = MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE;
		return MZ_RESULT_SUCCESS;
	}

	virtual mzResult ExecuteNode(const mzNodeExecuteArgs* args) {
		auto pinValues = GetPinValues(args);
		auto inputTextureInfo = DeserializeTextureInfo(pinValues[MZN_Input]);
		auto outputTextureInfo = DeserializeTextureInfo(pinValues[MZN_Output]);
		mzRunPassParams deinterlacePass = {};
		deinterlacePass.Key = MZN_Utilities_Deinterlace_Pass;
		auto field = inputTextureInfo.Info.Texture.FieldType;
		bool isInterlaced = IsTextureFieldTypeInterlaced(field);
		if (!isInterlaced) {
			mzEngine.LogW("Deinterlace Node: Input is not interlaced!");
			return MZ_RESULT_FAILED;
		}
		uint32_t isOdd = field - 1;
		std::vector bindings = {
			ShaderBinding(MZN_Input, inputTextureInfo),
			ShaderBinding(MZN_IsOdd, isOdd)
		};
		deinterlacePass.Bindings = bindings.data();
		deinterlacePass.BindingCount = bindings.size();
		deinterlacePass.Output = outputTextureInfo;
		deinterlacePass.DoNotClear = true;
		mzEngine.RunPass(0, &deinterlacePass);
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;
		outShaders[0] = { .Key = MZN_Utilities_Deinterlace_Fragment_Shader,
						 .Source = {.SpirvBlob = {(void*)Deinterlace_frag_spv, sizeof(Deinterlace_frag_spv)}} };
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes)
	{
		*count = 1;
		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes = mzPassInfo {
			.Key = MZN_Utilities_Deinterlace_Pass,
			.Shader = MZN_Utilities_Deinterlace_Fragment_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns)
	{
		*count = 0;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;
		return MZ_RESULT_SUCCESS;
	}
};

void RegisterInterlace(mzNodeFunctions* nodeFunctions)
{
	MZ_BIND_NODE_CLASS(MZN_TypeName_Utilities_Interlace, InterlaceNode, nodeFunctions);
}

void RegisterDeinterlace(mzNodeFunctions* nodeFunctions)
{
	MZ_BIND_NODE_CLASS(MZN_TypeName_Utilities_Deinterlace, DeinterlaceNode, nodeFunctions);
}

} // namespace mz::utilities
