#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosAI/nosAISubsystem.h"
#include "nosTensorSubsystem/nosTensorSubsystem.h"

NOS_INIT();
void RegisterONNXRunner(nosNodeFunctions* outFunctions);

extern nosAISubsystem* nosAI = nullptr;
extern nosTensorSubsystem* nosTensor = nullptr;
extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
	{
		*outCount = (size_t)(1);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		nosResult returnRes;
		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_AI_SUBSYSTEM_NAME), 1, 0, (void**)&nosAI);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_TENSOR_SUBSYSTEM_NAME), 1, 0, (void**)&nosTensor);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;
		
		RegisterONNXRunner(outFunctions[0]);

		return NOS_RESULT_SUCCESS;
	}
}