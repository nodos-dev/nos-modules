#pragma once
#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <limits>
#include <typeinfo>
#include<typeindex>

class nosTensor
{
public:
	nosTensor() : sizeOfSingleElement(0), type(nos::fb::TensorElementType::UNDEFINED), data(nullptr) {

	};
	nosTensor(int64_t channel, int64_t width, int64_t height) : sizeOfSingleElement(0), type(nos::fb::TensorElementType::UNDEFINED), data(nullptr) {
		Shape = std::vector<int64_t>{ 1, channel, width, height };
	}

	~nosTensor() {
		if (data != nullptr) {
			DataCleaner(data);
		}
	}

	void SetType(ONNXTensorElementDataType tensorDataType) {
		switch (tensorDataType) {
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
			type = (nos::fb::TensorElementType)tensorDataType;
			break;
		default:
			type = nos::fb::TensorElementType::UNDEFINED;
		}
	}

	void SetType(nos::fb::TensorElementType tensorDataType) {
		type = tensorDataType;
	}

	nos::fb::TensorElementType GetType() const {
		return type;
	}

	//void SetShape(std::vector<int64_t>&& shape ) {
	//	Shape = std::move(shape);
	//}

	void SetShape(std::vector<int32_t> shape) {
		Shape.clear();
		Shape.insert(Shape.begin(), shape.begin(), shape.end());
	}

	void SetShape(std::vector<int64_t> shape) {
		Shape.clear();
		Shape.insert(Shape.begin(), shape.begin(), shape.end());
	}
	void SetData(Ort::Value&& data) {
		Value = std::move(data);
	}

	int64_t GetLength() const {
		int64_t temp = 1;
		for (const auto& size : Shape) {
			temp *= size;
		}
		return temp;
	}

	std::string GetShapeStr() {
		std::string shapeStr;
		for (int i = 0; i < Shape.size(); i++) {
			shapeStr += std::to_string(Shape[i]);
			if (i < Shape.size() - 1) {
				shapeStr += "x";
			}
		}
		return shapeStr;
	}

	nos::fb::TTensor GetNativeTensor() {
		nos::fb::TTensor native;
		native.shape = Shape;
		native.type = type;
		native.buffer = reinterpret_cast<uint64_t>(data);
		return native;
	}

	void ApplySoftmax() {
		/*int64_t length = GetLength();
		float MAX = -INFINITY;
		for (int i = 0; i < length; i++) {
			if (*(dummyData.get() + i) > MAX) {
				MAX = *(dummyData.get() + i);
			}
		}
		std::vector<float> y(length);
		float sum = 0.0f;
		for (int i = 0; i < length; i++) {
			sum += y[i] = std::exp(*(dummyData.get() + i) - MAX);
		}
		for (int i = 0; i < length; i++) {
			*(dummyData.get() + i) = y[i] / sum;
		}*/
	}

	void SetTensorData(nos::fb::TensorElementType _type, int64_t p_data, int64_t count) {
		switch (type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return;
		case nos::fb::TensorElementType::UINT8:
			return SetTensorData<uint8_t>(reinterpret_cast<uint8_t*>(p_data), count);
		case nos::fb::TensorElementType::UINT16:
			return SetTensorData<uint16_t>(reinterpret_cast<uint16_t*>(p_data), count);
		case nos::fb::TensorElementType::UINT32:
			return SetTensorData<uint32_t>(reinterpret_cast<uint32_t*>(p_data), count);
		case nos::fb::TensorElementType::UINT64:
			return SetTensorData<uint64_t>(reinterpret_cast<uint64_t*>(p_data), count);
		case nos::fb::TensorElementType::INT8:
			return SetTensorData<int8_t>(reinterpret_cast<int8_t*>(p_data), count);
		case nos::fb::TensorElementType::INT16:
			return SetTensorData<int16_t>(reinterpret_cast<int16_t*>(p_data), count);
		case nos::fb::TensorElementType::INT32:
			return SetTensorData<int32_t>(reinterpret_cast<int32_t*>(p_data), count);
		case nos::fb::TensorElementType::INT64:
			return SetTensorData<int64_t>(reinterpret_cast<int64_t*>(p_data), count);
		case nos::fb::TensorElementType::FLOAT:
			return SetTensorData<float>(reinterpret_cast<float*>(p_data), count);
		case nos::fb::TensorElementType::FLOAT16:
			return SetTensorData<float>(reinterpret_cast<float*>(p_data), count);
		case nos::fb::TensorElementType::DOUBLE:
			return SetTensorData<double>(reinterpret_cast<double*>(p_data), count);
		case nos::fb::TensorElementType::BOOL:
			return SetTensorData<bool>(reinterpret_cast<bool*>(p_data), count);
		case nos::fb::TensorElementType::STRING:
			//SetPinValues<std::string>();
			break;
		}
		return;
	}

	//Does not copies or anything else, pure SET
	template <typename T>
	void SetTensorData(T* p_data, int64_t count) {
		DeduceType<T>();
		int length = GetLength();
		assert(count == length);
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
	}


	//TODO: Make one for GPU
	template <typename T>
	void CreateTensor(T* p_data, int64_t count, bool shouldCopy = false) {

		nos::fb::TensorElementType cachedType = type;

		DeduceType<T>();

		if (type == nos::fb::TensorElementType::UNDEFINED) {
			nosEngine.LogE("Can not create tensor with elements type of %s", typeid(T).name());
			return;
		}

		if (cachedType != type) {
			nosEngine.LogW("Tensor element type %s is different from expected.", typeid(T).name());
		}

		int length = GetLength();
		assert(count == length);

		if (shouldCopy) {
			if (data == nullptr) {
				CreateEmpty();
			}

			memcpy(data, p_data, count);

			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
			Value = Ort::Value::CreateTensor<T>(memory_info, static_cast<T*>(data), count, Shape.data(), Shape.size());
			sizeOfSingleElement = sizeof(T);
		}
		else {
			if (data != nullptr)
				DataCleaner(data);
			//If not copying, we should act as if we have the ownership
			data = p_data;
			DataCleaner = [](void* ptr) {delete[] static_cast<T*>(ptr); };
			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
			Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
			sizeOfSingleElement = sizeof(T);
		}
	}

	nosResult CreateEmpty() {
		switch (type) {
		case nos::fb::TensorElementType::UNDEFINED:
			return NOS_RESULT_FAILED;
		case nos::fb::TensorElementType::UINT8:
			return CreateEmpty<uint8_t>();
		case nos::fb::TensorElementType::UINT16:
			return CreateEmpty<uint16_t>();
		case nos::fb::TensorElementType::UINT32:
			return CreateEmpty<uint32_t>();
		case nos::fb::TensorElementType::UINT64:
			return CreateEmpty<uint64_t>();
		case nos::fb::TensorElementType::INT8:
			return CreateEmpty<int8_t>();
		case nos::fb::TensorElementType::INT16:
			return CreateEmpty<int16_t>();
		case nos::fb::TensorElementType::INT32:
			return CreateEmpty<int32_t>();
		case nos::fb::TensorElementType::INT64:
			return CreateEmpty<int64_t>();
		case nos::fb::TensorElementType::FLOAT:
			return CreateEmpty<float>();
		case nos::fb::TensorElementType::FLOAT16:
			return CreateEmpty<float>();
		case nos::fb::TensorElementType::DOUBLE:
			return CreateEmpty<double>();
		case nos::fb::TensorElementType::BOOL:
			return CreateEmpty<bool>();
		case nos::fb::TensorElementType::STRING:
			//SetPinValues<std::string>();
			break;
		}
		return NOS_RESULT_FAILED;
	}

	template <typename T>
	nosResult CreateEmpty() {
		int64_t length = GetLength();
		return AllocateMemory<T>(length);
	}

	template <typename T>
	nosResult AllocateMemory(size_t count) {
		if (data != nullptr) {
			DataCleaner(data);
		}
		data = new T[count];
		DataCleaner = [](void* ptr) {delete[] static_cast<T*>(ptr); };
		DeduceType<T>();
		sizeOfSingleElement = sizeof(T);
		return NOS_RESULT_SUCCESS;
	}

	Ort::Value* GetORTValuePointer() {
		return &Value;
	}

	const void* GetRawDataPointer() {
		return Value.GetTensorRawData();
	}

	//Can be used for memcpy operations?
	size_t GetSizeOfSingleElement() {
		return sizeOfSingleElement;
	}

	std::vector<int64_t> GetShape() {
		return Shape;
	}

	int64_t* GetShapePointer() {
		return Shape.data();
	}

	std::string GetNOSNameFromType() {
		switch (type) {
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
			return std::string("float");
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

	template<typename T>
	void DeduceType() {
		if (typeid(T) == typeid(uint8_t)) {
			type = nos::fb::TensorElementType::UINT8;
		}
		else if (typeid(T) == typeid(uint16_t)) {
			type = nos::fb::TensorElementType::UINT16;
		}
		else if (typeid(T) == typeid(uint32_t)) {
			type = nos::fb::TensorElementType::UINT32;
		}
		else if (typeid(T) == typeid(uint64_t)) {
			type = nos::fb::TensorElementType::UINT64;
		}
		else if (typeid(T) == typeid(int8_t)) {
			type = nos::fb::TensorElementType::INT8;
		}
		else if (typeid(T) == typeid(int16_t)) {
			type = nos::fb::TensorElementType::INT16;
		}
		else if (typeid(T) == typeid(int32_t)) {
			type = nos::fb::TensorElementType::INT32;
		}
		else if (typeid(T) == typeid(int64_t)) {
			type = nos::fb::TensorElementType::INT64;
		}
		else if (typeid(T) == typeid(std::string)) {
			type = nos::fb::TensorElementType::STRING;
		}
		else if (typeid(T) == typeid(float)) {
			type = nos::fb::TensorElementType::FLOAT;
		}
		else if (typeid(T) == typeid(double)) {
			type = nos::fb::TensorElementType::DOUBLE;
		}
		else if (typeid(T) == typeid(bool)) {
			type = nos::fb::TensorElementType::BOOL;
		}
		else {
			type = nos::fb::TensorElementType::UNDEFINED;
		}
	}

private:
	Ort::Value Value{ nullptr };
	nos::fb::TensorElementType type{ nos::fb::TensorElementType::UNDEFINED };
	std::vector<int64_t> Shape;
	void* data;
	size_t sizeOfSingleElement{ 0 }; //the sizeof(T)
	std::function<void(void*)> DataCleaner;
};

