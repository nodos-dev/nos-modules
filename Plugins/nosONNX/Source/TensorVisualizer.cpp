#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(TensorVisualizer);
NOS_REGISTER_NAME(In);

struct TensorVisualizerNodeContext : nos::NodeContext {
	nosUUID NodeID;
	nos::fb::TTensor TensorProxy;
	nosTensor nosInputTensor;
	std::vector<nosUUID> outputPins;
	nos::fb::TensorElementType type;
	TensorVisualizerNodeContext(nos::fb::Node const* node) :NodeContext(node) {
		NodeID = *node->id();
	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if (pinName.Compare(NSN_In.AsCStr()) == 0) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			TensorProxy = *tensor->UnPack();
			nosInputTensor.SetShape(TensorProxy.shape);
			nosInputTensor.SetType(TensorProxy.type);
			DeduceTypeAndPropogate(TensorProxy.type);
		}
	}

	void DeduceTypeAndPropogate(nos::fb::TensorElementType _type) {
		if (type != _type)
			ClearAllOutPins();

		switch (_type) {
		case nos::fb::TensorElementType::UNDEFINED:
			type = _type;
			nosEngine.LogE("Can,t visualize UNDEFINED typed tensor data!");
			break;
		case nos::fb::TensorElementType::UINT8:
			type = _type;
			SetPinValues<uint8_t>();
			break;
		case nos::fb::TensorElementType::UINT16:
			type = _type;
			SetPinValues<uint16_t>();
			break;
		case nos::fb::TensorElementType::UINT32:
			type = _type;
			SetPinValues<uint32_t>();
			break;
		case nos::fb::TensorElementType::UINT64:
			type = _type;
			SetPinValues<uint64_t>();
			break;
		case nos::fb::TensorElementType::INT8:
			type = _type;
			SetPinValues<int8_t>();
			break;
		case nos::fb::TensorElementType::INT16:
			type = _type;
			SetPinValues<int16_t>();
			break;
		case nos::fb::TensorElementType::INT32:
			type = _type;
			SetPinValues<int32_t>();
			break;
		case nos::fb::TensorElementType::INT64:
			type = _type;
			SetPinValues<int64_t>();
			break;
		case nos::fb::TensorElementType::FLOAT:
			type = _type;
			SetPinValues<float>();
			break;
		case nos::fb::TensorElementType::FLOAT16:
			type = _type;
			SetPinValues<float>();
			break;
		case nos::fb::TensorElementType::DOUBLE:
			type = _type;
			SetPinValues<double>();
			break;
		case nos::fb::TensorElementType::BOOL:
			type = _type;
			SetPinValues<bool>();
			break;
		case nos::fb::TensorElementType::STRING:

			//SetPinValues<std::string>();
			break;
		}
	}

	template<typename T>
	void SetPinValues() {
		T* data = reinterpret_cast<T*>(TensorProxy.buffer);
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
			std::string typeName = GetNOSNameFromType(type);
			for (int i = cachedSize; i < tensorLength; i++) {
				auto buf = nos::Buffer::From(*(data + i));
				auto pinData = std::vector<uint8_t>((uint8_t*)buf.data(), (uint8_t*)buf.data() + buf.size());
				outputPins.push_back(*(nosUUID*)generator.Generate()().as_bytes().data());
				CreatedOutputPins.push_back(nos::fb::CreatePinDirect(fbb,
					&outputPins.back(),
					std::string("FlatTensor_" + std::to_string(i)).c_str(),
					typeName.c_str(),
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

	void ClearAllOutPins() {
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<nos::fb::UUID> pinsToDelete;
		pinsToDelete.insert(pinsToDelete.begin(), outputPins.begin(), outputPins.end());
		outputPins.erase(outputPins.begin(), outputPins.end());
		HandleEvent(nos::CreateAppEvent(fbb,
			nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::CLEAR_PINS, &pinsToDelete)));
	}

	std::string GetNOSNameFromType(nos::fb::TensorElementType _type) {
		switch (_type) {
			case nos::fb::TensorElementType::UNDEFINED:
				return std::string("Undefined");
			case nos::fb::TensorElementType::UINT8:
				return std::string("ubyte");
			case nos::fb::TensorElementType::UINT16:
				return std::string("ushort");
			case nos::fb::TensorElementType::UINT32:
				return std::string("uint");
			case nos::fb::TensorElementType::UINT64:
				return std::string("ulong");
			case nos::fb::TensorElementType::INT8:
				return std::string("byte");
			case nos::fb::TensorElementType::INT16:
				return std::string("short");
			case nos::fb::TensorElementType::INT32:
				return std::string("int");
			case nos::fb::TensorElementType::INT64:
				return std::string("long");
			case nos::fb::TensorElementType::FLOAT:
				return std::string("float");
			case nos::fb::TensorElementType::FLOAT16:
				return std::string("");
			case nos::fb::TensorElementType::DOUBLE:
				return std::string("double");
			case nos::fb::TensorElementType::BOOL:
				return std::string("bool");
			case nos::fb::TensorElementType::STRING:
				//SetPinValues<std::string>();
				break;
		}
		return std::string("Undefined");

	}
};

void RegisterTensorVisualizer(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorVisualizer, TensorVisualizerNodeContext, outFunctions);
}