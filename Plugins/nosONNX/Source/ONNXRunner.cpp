#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosAI/nosAISubsystem.h"
#include "flatbuffers/flatbuffers.h"
#include "ORT_generated.h"

NOS_REGISTER_NAME(ONNXRunner);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(ModelPath);

struct ONNXRunnerNodeContext : nos::NodeContext
{
	nosUUID NodeUUID, InputUUID, OutputUUID;
	ONNXModel Model;
	ONNXRunnerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		NodeUUID = *node->id();
	}

	~ONNXRunnerNodeContext() {
	}

	void RunModel() {

	}

	nosResult LoadModel(std::filesystem::path modelPath, bool isReload = false) {

		ONNXModel mnistModel;
		ONNXLoadConfig loadConfig = {};
		loadConfig.OptimizationLevel = OPTIMIZATION_ENABLE_ALL;
		loadConfig.RunLocation = RUN_ONNX_ON_CUDA;
		std::filesystem::path optModelPath = modelPath.string() + "_optimized.ort";
		std::filesystem::path optModelJsonPath = optModelPath.string() + ".json";
		std::wstring optPath = optModelPath.wstring();
		loadConfig.OptimizedModelSavePath = optPath.c_str();
		ONNXCUDAOptions cudaOptions = { .DeviceID = 0 };
		loadConfig.CUDAOptions = &cudaOptions;

		nosResult res = nosAI->LoadONNXModel(&mnistModel, modelPath.string().c_str(), loadConfig);
		nosTensorInfo inputTensor = {};
		memcpy(&inputTensor.CreateInfo.ShapeInfo, &mnistModel.Inputs->Shape, sizeof(nosTensorShapeInfo));
		inputTensor.CreateInfo.Location = MEMORY_LOCATION_CUDA;
		inputTensor.CreateInfo.ElementType = mnistModel.Inputs[0].ElementType;

		nosTensorInfo outputTensor = {};
		memcpy(&outputTensor.CreateInfo.ShapeInfo, &mnistModel.Outputs->Shape, sizeof(nosTensorShapeInfo));
		outputTensor.CreateInfo.Location = MEMORY_LOCATION_CUDA;
		outputTensor.CreateInfo.ElementType = mnistModel.Outputs[0].ElementType;

		TensorPinConfig inputCfg = { .Name = "Input Tensor", .ShowAs = TENSOR_SHOW_AS_INPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_INPUT_PIN_OR_PROPERTY };
		TensorPinConfig outputCfg = { .Name = "Output Tensor", .ShowAs = TENSOR_SHOW_AS_OUTPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_OUTPUT_PIN_OR_PROPERTY };
		nosTensor->CreateTensorPin(&inputTensor, &NodeUUID, &InputUUID, inputCfg);
		nosTensor->CreateTensorPin(&outputTensor, &NodeUUID, &OutputUUID, outputCfg);

		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
	{
		*count = 1;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;
		names[0] = NOS_NAME_STATIC("LoadModel");
		fns[0] = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs) {
			if (ONNXRunnerNodeContext* onnxNode = static_cast<ONNXRunnerNodeContext*>(ctx))
			{
				//TODO: there may exists more than one inputs so be careful about index 0 here!
				auto values = nos::GetPinValues(nodeArgs);
				std::filesystem::path modelPath = GetPinValue<const char>(values, NSN_ModelPath);
				if (onnxNode->LoadModel(modelPath) != NOS_RESULT_SUCCESS) {
					nosEngine.LogE("Model load failed!");
				}
				
			}
		};
		return NOS_RESULT_SUCCESS;
	}
};

void RegisterONNXRunner(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_ONNXRunner, ONNXRunnerNodeContext, outFunctions);
}