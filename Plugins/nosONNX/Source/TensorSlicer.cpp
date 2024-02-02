#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(TensorSlicer);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(SliceFrom);

struct TensorSlicerNodeContext : nos::NodeContext {
	
	nos::fb::UUID NodeID, InputID, SliceDimensionID;
	nos::fb::TTensor InputTensorProxy;
	nosTensor nosInputTensor;
	std::vector<nos::fb::UUID> outputPins;
	std::vector<nos::fb::TTensor> OutputTensorProxies;
	std::vector<nosTensor> nosOutputTensors;
	nos::fb::TensorElementType type;
	std::vector<void*> SlicedData;

	size_t SliceIndex = 0;
	int64_t currentBuffer = 0;

	TensorSlicerNodeContext(nos::fb::Node const* node) : NodeContext(node) {
		NodeID = *node->id();

	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if (pinName.Compare(NSN_In.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			
			auto newProxy = *tensor->UnPack();
			bool tensorChanged = newProxy.shape.size() != InputTensorProxy.shape.size();
			bool dimensionsChanged = newProxy.shape.size() != InputTensorProxy.shape.size();
			for (int i = 0; i < newProxy.shape.size(); i++) {
				if (!tensorChanged && newProxy.shape[i] != InputTensorProxy.shape[i]) {
					tensorChanged = true;
					break;
				}
			}

			if (tensorChanged) {
				InputTensorProxy = newProxy;
				if (dimensionsChanged) {
					std::vector<std::string> list;
					
					for (int i = 0; i < newProxy.shape.size(); i++) {
						list.push_back(std::string("Dimension ") + std::to_string(i));
					}

					flatbuffers::FlatBufferBuilder fbb;
					flatbuffers::FlatBufferBuilder fbb2;
					std::vector<flatbuffers::Offset<nos::fb::Pin>> SliceFromPin;
					nos::fb::TVisualizer vis = { .type = nos::fb::VisualizerType::COMBO_BOX, .name = NSN_SliceFrom.AsString() };
					UUIDGenerator generator;

					auto buf = std::vector<u8>((u8*)list.front().data(), (u8*)list.front().data() + list.front().size() + 1);

					SliceDimensionID = *(nosUUID*)generator.Generate()().as_bytes().data();
					SliceFromPin.push_back(nos::fb::CreatePinDirect(fbb,
						&SliceDimensionID,
						NSN_SliceFrom.AsCStr(),
						"string",
						nos::fb::ShowAs::PROPERTY,
						nos::fb::CanShowAs::PROPERTY_ONLY,
						0,
						nos::fb::Visualizer::Pack(fbb, &vis), 
						&buf));

					HandleEvent(nos::CreateAppEvent(fbb,
						nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, &SliceFromPin)));

					HandleEvent(nos::CreateAppEvent(
						fbb2, nos::app::CreateUpdateStringList(fbb2, nos::fb::CreateStringList(fbb2, fbb2.CreateString(NSN_SliceFrom.AsString()), fbb2.CreateVectorOfStrings(list)))));
				}
			}

			if (currentBuffer != InputTensorProxy.buffer) {
				DeduceTypeAndPropogate(InputTensorProxy);
				currentBuffer = InputTensorProxy.buffer;
			}
			

			//nosInputTensor.SetShape(InputTensorProxy.shape);
			//nosInputTensor.SetType(InputTensorProxy.type);
			//DeduceTypeAndPropogate(InputTensorProxy);
			//nosInputTensor.CreateEmpty();
		}
		if (SliceDimensionID == pinId) {
			auto dimensionStr = std::string(static_cast<char*>(value->Data));
			int tokenIdx = dimensionStr.find(" ");
			SliceIndex = std::stoi(dimensionStr.substr(tokenIdx, std::string::npos));
			if (currentBuffer != 0) {
				DeduceTypeAndPropogate(InputTensorProxy);
				currentBuffer = InputTensorProxy.buffer;
			}
		}

	}

	void DeduceTypeAndPropogate(nos::fb::TTensor& proxy) {
		switch (proxy.type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return;
		case nos::fb::TensorElementType::UINT8:
			return Slice<uint8_t>(proxy);
		case nos::fb::TensorElementType::UINT16:
			return Slice<uint16_t>(proxy);
		case nos::fb::TensorElementType::UINT32:
			return Slice<uint32_t>(proxy);
		case nos::fb::TensorElementType::UINT64:
			return Slice<uint64_t>(proxy);
		case nos::fb::TensorElementType::INT8:
			return Slice<int8_t>(proxy);
		case nos::fb::TensorElementType::INT16:
			return Slice<int16_t>(proxy);
		case nos::fb::TensorElementType::INT32:
			return Slice<int32_t>(proxy);
		case nos::fb::TensorElementType::INT64:
			return Slice<int64_t>(proxy);
		case nos::fb::TensorElementType::FLOAT:
			return Slice<float>(proxy);
		case nos::fb::TensorElementType::FLOAT16:
			return Slice<float>(proxy);
		case nos::fb::TensorElementType::DOUBLE:
			return Slice<double>(proxy);
		case nos::fb::TensorElementType::BOOL:
			return Slice<bool>(proxy);
		case nos::fb::TensorElementType::STRING:
			//SetPinValues<std::string>(proxy);
			break;
		}
		return;
	}

	template <typename T>
	void Slice(nos::fb::TTensor const &proxy) {
		//TODO: use this for slicing tensor where data is pointer:
		// tensor_k = ( data + k*(MultiplicationOfRestOfDimensions), data + (k+1)*(MultiplicationOfRestOfDimensions) ) for k=0,dim-1
		T* incomingData = reinterpret_cast<T*>(proxy.buffer);
		std::vector<int64_t> newShape;
		int64_t offset = 1;
		size_t shapeLength = proxy.shape.size();
		size_t outputTensorCount = 0;
		for (int i = 0; i < shapeLength; i++) {
			if (i == SliceIndex) {
				outputTensorCount = proxy.shape[i];
				continue;
			}
			int64_t currentLength = proxy.shape[i];
			newShape.push_back(currentLength);
			offset *= currentLength;
		}

		OutputTensorProxies.clear();
		for (int i = 0; i < outputTensorCount; i++) {
			nos::fb::TTensor out;
			out.shape = newShape;
			out.buffer = proxy.buffer + i * offset;
			out.type = proxy.type;
			OutputTensorProxies.push_back(std::move(out));
		}
		CreateOutPins();
	}

	void CreateOutPins() {
		if (!outputPins.empty()) {
			ClearAllOutPins();
			outputPins.clear();
		}

		UUIDGenerator generator;
		flatbuffers::FlatBufferBuilder fbb;
		nos::fb::TPin thePin = {};
		
		std::vector<flatbuffers::Offset<nos::fb::Pin>> CreatedOutputPins;
		for (int i = 0; i < OutputTensorProxies.size(); i++) {
			auto buf = nos::Buffer::From(OutputTensorProxies[i]);
			auto pinData = std::vector<uint8_t>((uint8_t*)buf.data(), (uint8_t*)buf.data() + buf.size());
			outputPins.push_back(*(nosUUID*)generator.Generate()().as_bytes().data());
			CreatedOutputPins.push_back(nos::fb::CreatePinDirect(fbb,
				&outputPins.back(),
				std::string("Tensor " + std::to_string(i)).c_str(),
				nos::fb::Tensor::GetFullyQualifiedName(),
				nos::fb::ShowAs::OUTPUT_PIN,
				nos::fb::CanShowAs::OUTPUT_PIN_OR_PROPERTY,
				0,
				0,
				&pinData));
		}
		HandleEvent(nos::CreateAppEvent(fbb,
			nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, &CreatedOutputPins)));
	}

	void ClearAllOutPins() {
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<nos::fb::UUID> pinsToDelete;
		pinsToDelete.insert(pinsToDelete.begin(), outputPins.begin(), outputPins.end());
		outputPins.erase(outputPins.begin(), outputPins.end());
		HandleEvent(nos::CreateAppEvent(fbb,
			nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, &pinsToDelete)));
	}
};

void RegisterTensorSlicer(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorSlicer, TensorSlicerNodeContext, outFunctions);
}