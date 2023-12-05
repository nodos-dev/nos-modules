#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <onnxruntime_cxx_api.h>
#include "ONNXRTCommon.h"

NOS_REGISTER_NAME(ONNXRunner);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(ModelPath);

struct ONNXRunnerNodeContext : nos::NodeContext
{
	nosResourceShareInfo Input;   
	Ort::Env env;
	Ort::Session ModelSession{nullptr};

	nos::fb::TTensor nosInputTensor, nosOutputTensor;
	nosTensor InputTensor;
	nosTensor OutputTensor;
	std::string InputName, OutputName;
	nosUUID NodeID;


	ONNXRunnerNodeContext(nos::fb::Node const* node) :NodeContext(node)  {
		NodeID = *node->id();
	}

	void Run() {
		const char* input_names[] = { InputName.c_str() };
		const char* output_names[] = { OutputName.c_str() };

		ModelSession.Run(Ort::RunOptions{nullptr}, input_names, InputTensor.GetData(), 1, output_names, OutputTensor.GetData(), 1);
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
				if (std::filesystem::exists(modelPath)) {
					onnxNode->ModelSession = Ort::Session{ onnxNode->env, modelPath.c_str(), Ort::SessionOptions{nullptr} };
					onnxNode->InputTensor.SetShape(onnxNode->ModelSession.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
					onnxNode->OutputTensor.SetShape(onnxNode->ModelSession.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());

					int InputCount = onnxNode->ModelSession.GetInputCount();
					int OutputCount = onnxNode->ModelSession.GetOutputCount();

					onnxNode->nosInputTensor.shape = onnxNode->InputTensor.GetShape();
					onnxNode->nosOutputTensor.shape = onnxNode->OutputTensor.GetShape();

					flatbuffers::FlatBufferBuilder fbb_t;
					
					auto bufInput = nos::Buffer::From(onnxNode->nosInputTensor);
					auto inputTensorData = std::vector<uint8_t>((uint8_t*)bufInput.data(), (uint8_t*)bufInput.data() + bufInput.size());

					auto bufOutput = nos::Buffer::From(onnxNode->nosOutputTensor);
					auto outputTensorData = std::vector<uint8_t>((uint8_t*)bufOutput.data(), (uint8_t*)bufOutput.data() + bufOutput.size());

					std::vector<flatbuffers::Offset<nos::fb::Pin>> InputPins;
					std::vector<flatbuffers::Offset<nos::fb::Pin>> OutputPins;
					flatbuffers::FlatBufferBuilder fbb;
					flatbuffers::FlatBufferBuilder fbb2;
					

					std::optional<Ort::AllocatedStringPtr> inputName;
					std::optional<Ort::AllocatedStringPtr> outputName;
					Ort::AllocatorWithDefaultOptions ortAllocator;
					inputName.emplace(onnxNode->ModelSession.GetInputNameAllocated(0, ortAllocator));
					outputName.emplace(onnxNode->ModelSession.GetOutputNameAllocated(0, ortAllocator));

					onnxNode->InputName = {inputName->get()};
					onnxNode->OutputName = {outputName->get()};

					inputName->reset();
					outputName->reset();
					nosUUID id;
					UUIDGenerator generator;
					for (int i = 0; i < InputCount; i++) {
						inputName.emplace(onnxNode->ModelSession.GetInputNameAllocated(i, ortAllocator));
						InputPins.push_back(nos::fb::CreatePinDirect(fbb,
																(nosUUID*)generator.Generate()().as_bytes().data(),
																inputName->get(),
																"nos.fb.Tensor",
																nos::fb::ShowAs::INPUT_PIN,
																nos::fb::CanShowAs::INPUT_PIN_ONLY,
																0,
																0,
																&inputTensorData));
						inputName->reset();
					}

					for (int i = 0; i < OutputCount; i++) {
						outputName.emplace(onnxNode->ModelSession.GetOutputNameAllocated(i, ortAllocator));

						OutputPins.push_back(nos::fb::CreatePinDirect(fbb2,
																(nosUUID*)generator.Generate()().as_bytes().data(), outputName->get(),
																"nos.fb.Tensor",
																nos::fb::ShowAs::OUTPUT_PIN,
																nos::fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
																0,
																0,
																&outputTensorData));

						outputName->reset();
					}

					HandleEvent(nos::CreateAppEvent(fbb,
								nos::CreatePartialNodeUpdateDirect(fbb, &onnxNode->NodeId, nos::ClearFlags::NONE, 0, &InputPins)));

					HandleEvent(nos::CreateAppEvent(fbb2,
								nos::CreatePartialNodeUpdateDirect(fbb2, &onnxNode->NodeId, nos::ClearFlags::NONE, 0, &OutputPins)));

				}
			}
		};
		return NOS_RESULT_SUCCESS;
	}
};

void RegisterONNXRunner(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_ONNXRunner, ONNXRunnerNodeContext, outFunctions);
}