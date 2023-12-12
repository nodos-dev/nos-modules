#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <onnxruntime_cxx_api.h>
#include <limits>
#include "ONNXRTCommon.h"
#include "Tensor.h"

NOS_REGISTER_NAME(TensorPreprocessor);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);

struct TensorPreprocessorNodeContext : nos::NodeContext {

	nosUUID NodeID, InputID, OutputID;
	nosTensor nosInputTensor;
	nos::fb::TTensor OutTensorProxy;

	//TODO: GET RID OF THIS
	float* normalizedData;
	size_t dataSize;

	TensorPreprocessorNodeContext(nos::fb::Node const* node) : NodeContext(node), normalizedData(nullptr) {
		NodeID = *node->id();
		for (auto pin : *node->pins()) {
			if (NSN_Out.Compare(pin->name()->c_str()) == 0) {
				OutputID = *pin->id();
			}
		}
	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) {
		if(pinName == NSN_In) {
			auto tensor = flatbuffers::GetRoot<nos::fb::Tensor>(value->Data);
			auto tensorProxy = tensor->UnPack();
			nosInputTensor.SetShape(tensorProxy->shape);
			if (nosInputTensor.GetLength() != dataSize) {
				dataSize = nosInputTensor.GetLength();
				if (normalizedData != nullptr) {
					delete[] normalizedData;
				}
				normalizedData = new float[dataSize];
				
			}
			DeduceTypeAndPropogate(*tensorProxy);

			nos::fb::TTensor outTensor;
			outTensor.buffer = reinterpret_cast<int64_t>(normalizedData);
			outTensor.type = nos::fb::TensorElementType::FLOAT;
			outTensor.shape = tensorProxy->shape;
			nosEngine.SetPinValue(OutputID, nos::Buffer::From(outTensor));
		}
	}

	void DeduceTypeAndPropogate(nos::fb::TTensor& proxy) {
		switch (proxy.type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return;
		case nos::fb::TensorElementType::UINT8:
			return NormalizeData<uint8_t>(proxy.buffer);
		case nos::fb::TensorElementType::UINT16:
			return NormalizeData<uint16_t>(proxy.buffer);
		case nos::fb::TensorElementType::UINT32:
			return NormalizeData<uint32_t>(proxy.buffer);
		case nos::fb::TensorElementType::UINT64:
			return NormalizeData<uint64_t>(proxy.buffer);
		case nos::fb::TensorElementType::INT8:
			return NormalizeData<int8_t>(proxy.buffer);
		case nos::fb::TensorElementType::INT16:
			return NormalizeData<int16_t>(proxy.buffer);
		case nos::fb::TensorElementType::INT32:
			return NormalizeData<int32_t>(proxy.buffer);
		case nos::fb::TensorElementType::INT64:
			return NormalizeData<int64_t>(proxy.buffer);
		case nos::fb::TensorElementType::FLOAT:
			return NormalizeData<float>(proxy.buffer);
		case nos::fb::TensorElementType::FLOAT16:
			return NormalizeData<float>(proxy.buffer);
		case nos::fb::TensorElementType::DOUBLE:
			return NormalizeData<double>(proxy.buffer);
		case nos::fb::TensorElementType::BOOL:
			//return NormalizeData<bool>(proxy.buffer);
		case nos::fb::TensorElementType::STRING:
			//SetPinValues<std::string>(proxy);
			break;
		}
		return;
	}

	template<typename T>
	void NormalizeData(int64_t buffer) {
		T* data = reinterpret_cast<T*>(buffer);
		T max = std::numeric_limits<T>::max();
		
		for (int i = 0; i < dataSize; i++) {
			*(normalizedData + i) = 1.0f - (*(data + i)) / (static_cast<float>(std::numeric_limits<T>::max()));
		}
	}
};

void RegisterTensorPreprocessor(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_TensorPreprocessor, TensorPreprocessorNodeContext, outFunctions);
}