#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "nosAI/nosAI.h"
#include "flatbuffers/flatbuffers.h"
#include "ORT_generated.h"

NOS_REGISTER_NAME(ONNXRunner);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(ModelPath);

struct ONNXRunnerNodeContext : nos::NodeContext
{
	ONNXRunnerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		ONNXModel mnistModel;
		ONNXLoadConfig loadConfig = {};
		loadConfig.OptimizationLevel = OPTIMIZATION_ENABLE_BASIC;
		loadConfig.RunLocation = RUN_ONNX_ON_CPU;
		loadConfig.OptimizedModelSavePath = L"C:/WorkInParallel/Models/ONNX/";
		nosResult res = nosAI->LoadONNXModel(&mnistModel, "C:/WorkInParallel/Models/ONNX/mnist.onnx", loadConfig);
		
		std::ifstream infile;
		infile.open("C:/WorkInParallel/Models/ONNX/mnist.onnx_optimized.ort", std::ios::binary | std::ios::in);
		infile.seekg(0, std::ios::end);
		int length = infile.tellg();
		infile.seekg(0, std::ios::beg);
		char* data = new char[length];
		infile.read(data, length);
		infile.close();
		onnxruntime::fbs::TInferenceSession inferenceSession;
		auto ORTed = onnxruntime::fbs::GetInferenceSession(data);
		ORTed->UnPackTo(&inferenceSession);
		onnxruntime::fbs::TModel modelT;
		auto a = ORTed->GetFullyQualifiedName();
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