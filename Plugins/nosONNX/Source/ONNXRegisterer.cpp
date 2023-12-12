#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>

NOS_INIT();
void RegisterONNXRunner(nosNodeFunctions* outFunctions);
void RegisterRGBAtoTensor(nosNodeFunctions* outFunctions);
void RegisterTensorVisualizer(nosNodeFunctions* outFunctions);
void RegisterTensorSlicer(nosNodeFunctions* outFunctions);
void RegisterTensorPreprocessor(nosNodeFunctions* outFunctions);
void RegisterTensorToRGBA(nosNodeFunctions* outFunctions);

extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
	{
		*outCount = (size_t)(6);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		RegisterONNXRunner(outFunctions[0]);
		RegisterRGBAtoTensor(outFunctions[1]);
		RegisterTensorVisualizer(outFunctions[2]);
		RegisterTensorSlicer(outFunctions[3]);
		RegisterTensorPreprocessor(outFunctions[4]);
		RegisterTensorToRGBA(outFunctions[5]);

		return NOS_RESULT_SUCCESS;
	}
}