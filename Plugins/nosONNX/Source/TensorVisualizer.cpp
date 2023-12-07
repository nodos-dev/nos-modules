#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "ONNXRTCommon.h"

NOS_REGISTER_NAME(TensorVisualizer);
NOS_REGISTER_NAME(In);

struct TensorVisualizerNodeContext : nos::NodeContext {
	nosUUID NodeID;
	nos::fb::TTensor TensorProxy;
	nosTensor nosInputTensor;
	std::vector<nosUUID> outputPins;
	TensorVisualizerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		NodeID = *node->id();
	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if (pinName.Compare(NSN_In.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			TensorProxy = *tensor->UnPack();
			nosInputTensor.SetShape(TensorProxy.shape);
			float* data = reinterpret_cast<float*>(TensorProxy.buffer);
			if (data == nullptr) {
				nosEngine.LogE("No data to visualize!");
				return;
			}
			int tensorLength = nosInputTensor.GetLength();
			//TODO: The most obvious problem is that we need to delete created pins correctly and efficiently
			if (outputPins.size() < tensorLength) {
				int cachedSize = outputPins.size();
				//First refresh old values
				for (int i = 0; i < cachedSize; i++) {
					nosEngine.SetPinValue(outputPins[i], nos::Buffer::From(*(data + i)));
				}

				UUIDGenerator generator;
				flatbuffers::FlatBufferBuilder fbb;
				std::vector<flatbuffers::Offset<nos::fb::Pin>> CreatedOutputPins;
				for (int i = cachedSize; i < tensorLength; i++) {
					auto buf = nos::Buffer::From(*(data + i));
					auto pinData = std::vector<uint8_t>((uint8_t*)buf.data(), (uint8_t*)buf.data() + buf.size());
					outputPins.push_back(*(nosUUID*)generator.Generate()().as_bytes().data());
					CreatedOutputPins.push_back(nos::fb::CreatePinDirect(fbb,
						&outputPins.back(),
						std::string("FlatTensor_"+std::to_string(i)).c_str(),
						"float", //TODO: Create a tensor storage type in fbs and use it!!!
						nos::fb::ShowAs::OUTPUT_PIN,
						nos::fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
						0,
						0,
						&pinData));
				}
				HandleEvent(nos::CreateAppEvent(fbb,
							nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, &CreatedOutputPins)));

			}
			else if (outputPins.size() > tensorLength) {

				int cachedSize = outputPins.size();
				for (int i = 0; i < tensorLength; i++) {
					nosEngine.SetPinValue(outputPins[i], nos::Buffer::From(*(data + i)));
				}

				flatbuffers::FlatBufferBuilder fbb;
				std::vector<nos::fb::UUID> pinsToDelete;
				pinsToDelete.insert(pinsToDelete.begin(), outputPins.begin() + tensorLength, outputPins.end());
				outputPins.erase(outputPins.begin() + tensorLength, outputPins.end());
				HandleEvent(nos::CreateAppEvent(fbb,
					nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::CLEAR_PINS, &pinsToDelete)));
			}
			else if (outputPins.size() == tensorLength) {
				int cachedSize = outputPins.size();
				for (int i = 0; i < tensorLength; i++) {
					nosEngine.SetPinValue(outputPins[i], nos::Buffer::From(*(data + i)));
				}
			}


		}
	}
};

void RegisterTensorVisualizer(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorVisualizer, TensorVisualizerNodeContext, outFunctions);
}