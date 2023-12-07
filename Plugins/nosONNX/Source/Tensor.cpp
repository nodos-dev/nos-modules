nosTensor::nosTensor(int64_t channel, int64_t width, int64_t height) {
	Shape = std::vector<int64_t>{ 1, channel, width, height };
}

void nosTensor::SetType(ONNXTensorElementDataType tensorDataType) {
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

nos::fb::TensorElementType nosTensor::GetType() const {
	return type;
}
void nosTensor::SetShape(int64_t channel, int64_t width, int64_t height)
{
	Shape = std::vector<int64_t>{ 1, channel, width, height };
}

//void SetShape(std::vector<int64_t>&& shape ) {
//	Shape = std::move(shape);
//}

void nosTensor::SetShape(std::vector<int32_t> shape) {
	Shape.clear();
	Shape.insert(Shape.begin(), shape.begin(), shape.end());
}

void nosTensor::SetShape(std::vector<int64_t> shape) {
	Shape.clear();
	Shape.insert(Shape.begin(), shape.begin(), shape.end());
}
void nosTensor::SetData(Ort::Value&& data) {
	Value = std::move(data);
}

int64_t nosTensor::GetLength() const {
	int64_t temp = 1;
	for (const auto& size : Shape) {
		temp *= size;
	}
	return temp;
}

void nosTensor::ApplySoftmax() {
	int64_t length = GetLength();
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
	}
}

//TODO: Make one for GPU
template <typename T>
void nosTensor::CreateTensor(T* p_data, int64_t count, bool shouldNormalize) {

	nos::fb::TensorElementType cachedType = type;

	DeduceType(*p_data);

	if (type == nos::fb::TensorElementType::UNDEFINED) {
		nosEngine.LogE("Can not create tensor with elements type of %s", typeid(T).name());
		return;
	}

	if (cachedType != type) {
		nosEngine.LogW("Tensor element type %s is different from expected.", typeid(T).name());
	}

	int length = GetLength();
	assert(count == length);
	//if (shouldNormalize) {
	//	
	//	if (rawData == nullptr) {
	//		rawData = std::make_unique<float[]>(count);
	//	}
	//	std::string log;
	//	for (int i = 0; i < count; i++) {
	//		log += std::string("   ");
	//		rawData[i] = 1.0f - ((float)*(p_data + 4*i)) / ((float) std::numeric_limits<T>::max());
	//		log += std::to_string((int)(rawData[i]));
	//		if (i % 28 == 0 ) {
	//			nosEngine.LogI("log: %s", log.c_str());
	//			log.clear();
	//		}
	//	}
	//	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	//	Value = Ort::Value::CreateTensor<float>(memory_info, rawData.get(), count, Shape.data(), Shape.size());
	//	auto returnedData = Value.GetTensorData<int>();
	//	return;
	//}
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
	sizeOfSingleElement = sizeof(T);
}

//Creates an empty tensor of floats with pre-set shape
void nosTensor::CreateEmpty() {
	int64_t length = GetLength();

	if (dummyData != nullptr)
		dummyData.reset();

	dummyData = std::make_unique<float[]>(length);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value = Ort::Value::CreateTensor<float>(memory_info, dummyData.get(), length, Shape.data(), Shape.size());
}

Ort::Value* nosTensor::GetORTValuePointer() {
	return &Value;
}

const void* nosTensor::GetRawData() {
	return Value.GetTensorRawData();
}

//Can be used for memcpy operations?
size_t nosTensor::GetSizeOfSingleElement() {
	return sizeOfSingleElement;
}

std::vector<int64_t> nosTensor::GetShape() {
	return Shape;
}

int64_t* nosTensor::GetShapePointer() {
	return Shape.data();
}

template<typename T>
void nosTensor::DeduceType(T param) {
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