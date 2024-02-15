// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include "Services.h"

NOS_INIT();

namespace nos::cudass
{
	extern "C"
	{

		NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions, void** exported)
		{
			nos::cudass::Initialize(0);
			auto subsystem = new nosCUDASubsystem;
			nos::cudass::Bind(subsystem);
			*exported = subsystem;
			return NOS_RESULT_SUCCESS;
		}

		NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystemTypeFunctions(size_t* outSize, nosSubsystemTypeFunctions** outList)
		{
			*outSize = 0;
			if (!outList)
				return NOS_RESULT_SUCCESS;
			return NOS_RESULT_SUCCESS;
		}

		NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystemNodeFunctions(size_t* outSize, nosSubsystemNodeFunctions** outList)
		{
			*outSize = 0;
			if (!outList)
				return NOS_RESULT_SUCCESS;
			return NOS_RESULT_SUCCESS;
		}

		NOSAPI_ATTR nosResult NOSAPI_CALL nosUnloadSubsystem(void* subsystemContext)
		{
			//TODO: Garbage Collect?
			return NOS_RESULT_SUCCESS;
		}
	}

}
