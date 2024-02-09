// Copyright MediaZ AS. All Rights Reserved.

#pragma once
#include "nosAI/nosAI.h"
#include "nosAIGlobals.h"
#include"AICommonMacros.h"

namespace nos::ai
{
	nosResult Bind(nosAISubsystem* subsys);
	nosResult NOSAPI_CALL LoadONNXModel(ONNXModel* model, const char* path, ONNXLoadConfig config);
	nosResult NOSAPI_CALL RunONNXModel(ONNXModel* model);
	nosResult NOSAPI_CALL SetONNXModelInput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);
	nosResult NOSAPI_CALL SetONNXModelOutput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);
}
