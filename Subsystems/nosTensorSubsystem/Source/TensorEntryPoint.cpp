// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include "TensorServices.h"
#include "nosTensorSubsystem/TensorTypes_generated.h"

NOS_INIT();

namespace nos::tensor
{
	NOS_REGISTER_NAME_SPACED(TensorTypeName, nos::sys::tensor::Tensor::GetFullyQualifiedName());
	extern "C"
	{

		NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions, void** exported)
		{
			auto subsystem = new nosTensorSubsystem;
			nosResult res = nos::tensor::Bind(subsystem);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			*exported = subsystem;
			return NOS_RESULT_SUCCESS;
		}

		NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystemTypeFunctions(size_t* outSize, nosSubsystemTypeFunctions** outList)
		{
			*outSize = 1;
			if (!outList)
				return NOS_RESULT_SUCCESS;
			auto textureTypeFunctions = outList[0];
			textureTypeFunctions->TypeName = NSN_TensorTypeName;
			textureTypeFunctions->CanConnectPins = nos::tensor::type::CanConnectPins;
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
