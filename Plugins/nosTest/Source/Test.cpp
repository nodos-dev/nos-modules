// Copyright Nodos AS. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/Helpers.hpp>

#include "GpuStress.frag.spv.dat"

NOS_INIT();
NOS_REGISTER_NAME(in1)
NOS_REGISTER_NAME(in2)
NOS_REGISTER_NAME(out)


static void TestFunction(void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
{
	auto args = nos::GetPinValues(functionArgs);

	auto a = *GetPinValue<double>(args, NSN_in1);
	auto b = *GetPinValue<double>(args, NSN_in2);
	auto c = a + b;
	nosEngine.SetPinValue(functionArgs->PinIds[2], {.Data = &c, .Size = sizeof(c)});
}

static nosResult GetFunctions(size_t* outCount, nosName* pName, nosPfnNodeFunctionExecute* fns)
{
	*outCount = 1;
	if (!pName || !fns)
		return NOS_RESULT_SUCCESS;

	*fns = TestFunction;
	*pName = NOS_NAME_STATIC("TestFunction");
	return NOS_RESULT_SUCCESS;
}


extern "C"
{

	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
	{
		*outCount = (size_t)(6);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;
		
		outFunctions[0]->TypeName = NOS_NAME_STATIC("nos.test.NodeTest");
		outFunctions[0]->GetFunctions = GetFunctions;
		outFunctions[1]->TypeName = NOS_NAME_STATIC("nos.test.NodeWithCategories");
		outFunctions[2]->TypeName = NOS_NAME_STATIC("nos.test.NodeWithFunctions");
		outFunctions[3]->TypeName = NOS_NAME_STATIC("nos.test.NodeWithCustomTypes");
		outFunctions[4]->TypeName = NOS_NAME_STATIC("nos.test.GpuStress");		
		outFunctions[4]->GetShaderSource = [](nosShaderSource* src) -> nosResult {
			src->SpirvBlob.Data = (void*)GpuStress_frag_spv;
			src->SpirvBlob.Size = sizeof(GpuStress_frag_spv);
			return NOS_RESULT_SUCCESS;
		};
		outFunctions[5]->TypeName = NOS_NAME_STATIC("nos.test.CopyTest");
		outFunctions[5]->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args)
		{
			nosCmd cmd;
			nosEngine.Begin(&cmd);
			auto values = nos::GetPinValues(args);
			nosResourceShareInfo input = nos::DeserializeTextureInfo(values[NOS_NAME_STATIC("Input")]);
			nosResourceShareInfo output = nos::DeserializeTextureInfo(values[NOS_NAME_STATIC("Output")]);
			nosEngine.Copy(cmd, &input, &output, 0);
			nosEngine.End(cmd);
			return NOS_RESULT_SUCCESS;
		};
		return NOS_RESULT_SUCCESS;
	}

}
