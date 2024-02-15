// Copyright MediaZ AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <cstring>
#include "nosAIServices.h"

extern nosVulkanSubsystem* nosVulkan = nullptr;
extern nosCUDASubsystem* nosCUDA = nullptr;

namespace nos::ai
{
	nosResult Bind(nosAISubsystem* subsys) {
		ONNX_Runner.Initialize(ONNXLogLevel::LOG_LEVEL_WARNING, "NOS AI Subsystem");
		subsys->LoadONNXModel = LoadONNXModel;
		subsys->RunONNXModel = RunONNXModel;
		subsys->SetONNXModelInput = SetONNXModelInput;
		subsys->SetONNXModelOutput = SetONNXModelOutput;
		return NOS_RESULT_SUCCESS;
	}
	nosResult NOSAPI_CALL nos::ai::LoadONNXModel(ONNXModel* model, const char* path, ONNXLoadConfig config)
	{
		return ONNX_Runner.LoadONNXModel(model, path, config);
	}
	nosResult NOSAPI_CALL RunONNXModel(ONNXModel* model)
	{
		return ONNX_Runner.RunONNXModel(model);
	}
	nosResult NOSAPI_CALL SetONNXModelInput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo)
	{
		return ONNX_Runner.SetModelInput(model, inputIndex, Data, memoryInfo);
	}
	nosResult NOSAPI_CALL SetONNXModelOutput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo)
	{
		return ONNX_Runner.SetModelOutput(model, inputIndex, Data, memoryInfo);
	}
}
