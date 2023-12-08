#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(TensorSlicer);
NOS_REGISTER_NAME(In);

struct TensorSlicerNodeContext : nos::NodeContext {
	
	nos::fb::UUID NodeID, InputID, SliceDimensionID;
	nos::fb::TTensor InputTensorProxy;
	nosTensor nosInputTensor;
	std::vector<nos::fb::UUID> outputPins;
	std::vector<nos::fb::TTensor> OutputTensorProxies;
	std::vector<nosTensor> nosOutputTensors;
	nos::fb::TensorElementType type;
	std::vector<void*> SlicedData;

	TensorSlicerNodeContext(nos::fb::Node const* node) : NodeContext(node) {
		NodeID = *node->id();

	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if (pinName.Compare(NSN_In.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			
			auto newProxy = *tensor->UnPack();
			bool tensorChanged = newProxy.shape.size() != InputTensorProxy.shape.size();
			bool dimensionsChanged = newProxy.shape.size() != InputTensorProxy.shape.size();
			for (int i = 0; i < newProxy.shape.size(); i++) {
				if (!tensorChanged && newProxy.shape[i] != InputTensorProxy.shape[i]) {
					tensorChanged = true;
				}
			}

			if (tensorChanged) {
				InputTensorProxy = newProxy;
				if (dimensionsChanged) {
					
					std::string sliceFrom("Slice From");
					std::vector<std::string> list;
					
					for (int i = 0; i < newProxy.shape.size(); i++) {
						list.push_back(std::string("Dimension") + std::to_string(i));
					}

					flatbuffers::FlatBufferBuilder fbb;
					flatbuffers::FlatBufferBuilder fbb2;
					std::vector<flatbuffers::Offset<nos::fb::Pin>> SliceFromPin;
					nos::fb::TVisualizer vis = { .type = nos::fb::VisualizerType::COMBO_BOX, .name = sliceFrom };
					UUIDGenerator generator;

					auto buf = std::vector<u8>((u8*)list.front().data(), (u8*)list.front().data() + list.front().size() + 1);

					SliceDimensionID = *(nosUUID*)generator.Generate()().as_bytes().data();
					SliceFromPin.push_back(nos::fb::CreatePinDirect(fbb,
						&SliceDimensionID,
						sliceFrom.c_str(),
						"string",
						nos::fb::ShowAs::PROPERTY,
						nos::fb::CanShowAs::PROPERTY_ONLY,
						0,
						nos::fb::Visualizer::Pack(fbb, &vis), 
						&buf));

					HandleEvent(nos::CreateAppEvent(fbb,
						nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, &SliceFromPin)));

					HandleEvent(nos::CreateAppEvent(
						fbb2, nos::app::CreateUpdateStringList(fbb2, nos::fb::CreateStringList(fbb2, fbb2.CreateString(sliceFrom), fbb2.CreateVectorOfStrings(list)))));
				}
			}
			

			//nosInputTensor.SetShape(InputTensorProxy.shape);
			//nosInputTensor.SetType(InputTensorProxy.type);
			//DeduceTypeAndPropogate(InputTensorProxy);
			//nosInputTensor.CreateEmpty();
		}
		if (pinId == SliceDimensionID) {
			
		}

	}

	void DeduceTypeAndPropogate(nos::fb::TTensor& proxy) {
		switch (type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return;
		case nos::fb::TensorElementType::UINT8:
			return SliceAndCopyData<uint8_t>(proxy);
		case nos::fb::TensorElementType::UINT16:
			return SliceAndCopyData<uint16_t>(proxy);
		case nos::fb::TensorElementType::UINT32:
			return SliceAndCopyData<uint32_t>(proxy);
		case nos::fb::TensorElementType::UINT64:
			return SliceAndCopyData<uint64_t>(proxy);
		case nos::fb::TensorElementType::INT8:
			return SliceAndCopyData<int8_t>(proxy);
		case nos::fb::TensorElementType::INT16:
			return SliceAndCopyData<int16_t>(proxy);
		case nos::fb::TensorElementType::INT32:
			return SliceAndCopyData<int32_t>(proxy);
		case nos::fb::TensorElementType::INT64:
			return SliceAndCopyData<int64_t>(proxy);
		case nos::fb::TensorElementType::FLOAT:
			return SliceAndCopyData<float>(proxy);
		case nos::fb::TensorElementType::FLOAT16:
			return SliceAndCopyData<float>(proxy);
		case nos::fb::TensorElementType::DOUBLE:
			return SliceAndCopyData<double>(proxy);
		case nos::fb::TensorElementType::BOOL:
			return SliceAndCopyData<bool>(proxy);
		case nos::fb::TensorElementType::STRING:
			//SetPinValues<std::string>(proxy);
			break;
		}
		return;
	}

	template <typename T>
	void SliceAndCopyData(nos::fb::TTensor const &proxy) {
		//TODO: use this for slicing tensor where data is pointer:
		// tensor_k = ( data + k*(MultiplicationOfRestOfDimensions), data + (k+1)*(MultiplicationOfRestOfDimensions) ) for k=0,dim-1
		//T* incomingData = reinterpret_cast<T*>(proxy.buffer);
		//
		//int tensorLength = nosInputTensor.GetLength();
		//std::vector<int> offsets;
		//for (int i = 0; i < proxy.shape.size(); i++) {
		//	int offset = (i == 0) ? (0) : 1;
		//	bool shouldAllocate = true;
		//	if (i < nosOutputTensors.size() && proxy.shape[i] == nosOutputTensors[i].GetLength() && SlicedData[i] != nullptr;) {
		//		shouldAllocate = false;
		//	}

		//	offsets.push_back();
		//	if (shouldAllocate) {
		//		SlicedData.push_back(new T[proxy.shape[i]);
		//		nosInputTensor tensor;
		//		
		//		nosOutputTensors.push_back(std::move(tensor));
		//	}
		//}

		//for (int i = 0; i < offsets.size(); i++) {
		//	int offset = offsets[i];
		//	for (int j = 0; j < tensorLength; j+=offset) {
		//		//Update the data
		//		SlicedData[i][(j % offset)] = *(incomingData + j);
		//	}
		//	if (nosOutputTensors[i].GetRawDataPointer() == nullptr) {
		//		nosOutputTensors[i].SetShape({ 1, offset });
		//		nosOutputTensors[i].SetType(proxy.type);
		//		nosOutputTensors[i].CreateTensor(slicedData[i], offset, false);
		//	}
		//}
	}

	void CreateOutPins(nos::fb::TTensor const& proxy) {
		UUIDGenerator generator;
		flatbuffers::FlatBufferBuilder fbb;

	}
};

void RegisterTensorSlicer(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorSlicer, TensorSlicerNodeContext, outFunctions);
}