#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <nosTensorSubsystem/TensorTypes_generated.h>
#include <nosAI/nosAISubsystem.h>
#include <flatbuffers/flatbuffers.h>
#include "ORT_generated.h"

NOS_REGISTER_NAME(ONNXRunner);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(ModelPath);

struct ONNXRunnerNodeContext : nos::NodeContext
{
	nosUUID NodeUUID, InputUUID, OutputUUID;
	ONNXModel Model;
	std::condition_variable RunModel_CV;
	std::mutex RunModelMutex;
	std::thread ModelRunnerThread;

	nosTensorInfo InputTensor = {}, OutputTensor = {};
	nosCUDABufferInfo OutputTensorCUDABuffer = {};
	bool IsFixedOutput = true;
	ONNXRunnerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		NodeUUID = *node->id();
	}

	~ONNXRunnerNodeContext() {
	}

	virtual void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) {
		if (InputUUID == pinId) {
			if (IsFixedOutput)
				return;
			if (OutputTensor.CreateInfo.ShapeInfo.Dimensions != NULL) {
				auto tensor = flatbuffers::GetRoot<nos::sys::tensor::Tensor>(value.Data);
				for (int i = 0; i < OutputTensor.CreateInfo.ShapeInfo.DimensionCount; i++) {
					if (OutputTensor.CreateInfo.ShapeInfo.Dimensions[i] == -1) {
						//THIS IS NOT ROBUST AT ALL, MUST HELD A BETTER ASSUMPTION
						OutputTensor.CreateInfo.ShapeInfo.Dimensions[i] = tensor->shape()->Get(i);
					}
				}
				PrepareOutput();
			}
			

		}
	}

	void PrepareOutput() {
		if (OutputTensor.CreateInfo.Location != MEMORY_LOCATION_CUDA)
			return;

		if (OutputTensorCUDABuffer.CreateInfo.AllocationSize == OutputTensor.MemoryInfo.Size) {
			return;
		}

		for (int i = 0; i < OutputTensor.CreateInfo.ShapeInfo.DimensionCount; i++) {
			if (OutputTensor.CreateInfo.ShapeInfo.Dimensions[i] == -1) {
				return;
			}
		}

		if (OutputTensorCUDABuffer.Address != NULL) {
			nosCUDA->DestroyBuffer(&OutputTensorCUDABuffer);
		}
		nosCUDA->CreateShareableBufferOnCUDA(&OutputTensorCUDABuffer, OutputTensor.MemoryInfo.Size);
		
		nosTensor->ImportTensorFromCUDABuffer(&OutputTensor, &OutputTensorCUDABuffer, OutputTensor.CreateInfo.ShapeInfo, OutputTensor.CreateInfo.ElementType);

		TensorPinConfig outputCfg = { .Name = "Output Tensor", .ShowAs = TENSOR_SHOW_AS_OUTPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_OUTPUT_PIN_OR_PROPERTY };
		nosTensor->UpdateTensorPin(&OutputTensor, &NodeUUID, &OutputUUID, outputCfg);
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override {

		return NOS_RESULT_SUCCESS; 
	}


	void RunModel() {

	}

	nosResult LoadModel(std::filesystem::path modelPath, bool isReload = false) {

		ONNXLoadConfig loadConfig = {};
		loadConfig.OptimizationLevel = OPTIMIZATION_ENABLE_ALL;
		loadConfig.RunLocation = RUN_ONNX_ON_CUDA;
		std::filesystem::path optModelPath = modelPath.string() + "_optimized.ort";
		std::filesystem::path optModelJsonPath = optModelPath.string() + ".json";
		std::wstring optPath = optModelPath.wstring();
		loadConfig.OptimizedModelSavePath = optPath.c_str();
		ONNXCUDAOptions cudaOptions = { .DeviceID = 0 };
		loadConfig.CUDAOptions = &cudaOptions;

		nosResult res = nosAI->LoadONNXModel(&Model, modelPath.string().c_str(), loadConfig);
		memcpy(&InputTensor.CreateInfo.ShapeInfo, &Model.Inputs->Shape, sizeof(nosTensorShapeInfo));
		InputTensor.CreateInfo.Location = MEMORY_LOCATION_CUDA;
		//TODO: Query for multiple inputs / outputs!!!
		InputTensor.CreateInfo.ElementType = Model.Inputs[0].ElementType;

		memcpy(&OutputTensor.CreateInfo.ShapeInfo, &Model.Outputs->Shape, sizeof(nosTensorShapeInfo));
		OutputTensor.CreateInfo.Location = MEMORY_LOCATION_CUDA;
		OutputTensor.CreateInfo.ElementType = Model.Outputs[0].ElementType;
		IsFixedOutput = true;
		for (int i = 0; i < OutputTensor.CreateInfo.ShapeInfo.DimensionCount; i++) {
			if (OutputTensor.CreateInfo.ShapeInfo.Dimensions[i] == -1) {
				IsFixedOutput = false;
			}
		}

		TensorPinConfig inputCfg = { .Name = "Input Tensor", .ShowAs = TENSOR_SHOW_AS_INPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_INPUT_PIN_OR_PROPERTY };
		TensorPinConfig outputCfg = { .Name = "Output Tensor", .ShowAs = TENSOR_SHOW_AS_OUTPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_OUTPUT_PIN_OR_PROPERTY };
		nosTensor->UpdateTensorPin(&InputTensor, &NodeUUID, &InputUUID, inputCfg);
		nosTensor->UpdateTensorPin(&OutputTensor, &NodeUUID, &OutputUUID, outputCfg);
		PrepareOutput();
		return NOS_RESULT_SUCCESS;
	}

	static nosBool CanConnectPin(void* ctx, nosName pinName, nosUUID connectedPinId, const nosBuffer* connectedPinData) {
		ONNXRunnerNodeContext* node = reinterpret_cast<ONNXRunnerNodeContext*>(ctx);
		const nos::sys::tensor::Tensor* connectedTensor = flatbuffers::GetRoot<nos::sys::tensor::Tensor>(connectedPinData->Data);
		nos::sys::tensor::TensorElementDataType ConnectedElementType = connectedTensor->element_type();
		if (ConnectedElementType != (nos::sys::tensor::TensorElementDataType)node->InputTensor.CreateInfo.ElementType) {
			return NOS_FALSE;
		}

		if (connectedTensor->shape()->size() != node->InputTensor.CreateInfo.ShapeInfo.DimensionCount) { 
			return NOS_FALSE; 
		}

		for (int i = 0; i < node->InputTensor.CreateInfo.ShapeInfo.DimensionCount; i++) {
			if (node->InputTensor.CreateInfo.ShapeInfo.Dimensions[i]!=-1 && connectedTensor->shape()->Get(i) != node->InputTensor.CreateInfo.ShapeInfo.Dimensions[i]) {
				return NOS_FALSE;
			}
		}

		return NOS_TRUE;
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
	outFunctions->CanConnectPin = ONNXRunnerNodeContext::CanConnectPin;
	NOS_BIND_NODE_CLASS(NSN_ONNXRunner, ONNXRunnerNodeContext, outFunctions);
}

