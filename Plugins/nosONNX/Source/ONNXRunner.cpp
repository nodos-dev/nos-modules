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

	ONNXRunnerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		NodeUUID = *node->id();
		ONNXModel mnistModel;
		ONNXLoadConfig loadConfig = {};
		loadConfig.OptimizationLevel = OPTIMIZATION_ENABLE_ALL;
		loadConfig.RunLocation = RUN_ONNX_ON_CPU;
		std::filesystem::path modelPath = "D:/AI-Models/ONNX/depth_anything_vits14.onnx";
		std::filesystem::path optModelPath = modelPath.string() + "_optimized.ort";
		std::filesystem::path optModelJsonPath = optModelPath.string() + ".json";
		std::wstring optPath = optModelPath.wstring();
		loadConfig.OptimizedModelSavePath = optPath.c_str();

		nosResult res = nosAI->LoadONNXModel(&mnistModel, modelPath.string().c_str(), loadConfig);
		nosTensorInfo inputTensor = {};
		memcpy(&inputTensor.CreateInfo.ShapeInfo, &mnistModel.Inputs->Shape, sizeof(nosTensorShapeInfo));
		inputTensor.CreateInfo.Location = MEMORY_LOCATION_CUDA;
		inputTensor.CreateInfo.ElementType = mnistModel.Inputs[0].ElementType;
		TensorPinConfig inputCfg = { .Name = "Input Tensor", .ShowAs = TENSOR_SHOW_AS_INPUT_PIN, .CanShowAs = TENSOR_CAN_SHOW_AS_INPUT_PIN_OR_PROPERTY };
		nosTensor->CreateTensorPin(&inputTensor, &NodeUUID, &InputUUID, inputCfg);

		std::ifstream infile;
		infile.open(optModelPath, std::ios::binary | std::ios::in);
		infile.seekg(0, std::ios::end);
		uint64_t length = infile.tellg();

		//infile.seekg(0, std::ios::beg);
		//char* data = new char[length];
		//infile.read(data, length);
		//infile.close();
		//onnxruntime::fbs::TInferenceSession inferenceSession;
		//auto ORTed = onnxruntime::fbs::GetInferenceSession(data);
		//ORTed->UnPackTo(&inferenceSession);
		//std::ifstream schemaFile;
		//schemaFile.open("C:/dev/onnxrt-modules/Plugins/nosONNX/Config/ORT.fbs", std::ios::binary | std::ios::in);
		//std::string content{ std::istreambuf_iterator<char>(schemaFile), std::istreambuf_iterator<char>() };
		//
		//// Close the file
		//schemaFile.close();
		//flatbuffers::Parser parser;
		//parser.Parse(content.c_str(), { nullptr });
		//std::string name = onnxruntime::fbs::InferenceSession::GetFullyQualifiedName();;
		//parser.SetRootType(name.c_str());
		//std::string out;
		//flatbuffers::GenText(parser, data, &out);
		//flatbuffers::SaveFile(optModelJsonPath.string().c_str(), out, false);
		//onnxruntime::fbs::TModel modelT;
		//auto a = ORTed->GetFullyQualifiedName();
	}

	~ONNXRunnerNodeContext() {
	}

	void RunModel() {

	}

	nosResult LoadModel(std::filesystem::path modelPath, bool isReload = false) {
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