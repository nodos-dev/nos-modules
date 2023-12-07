#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <limits>

struct nosTensor
{
public:
	nosTensor() = default;
	nosTensor(int64_t channel, int64_t width, int64_t height) {
		Shape = std::vector<int64_t>{1, channel, width, height};
	}

	void SetShape(int64_t channel, int64_t width, int64_t height) {
		Shape = std::vector<int64_t> {1, channel, width, height};
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

	int64_t GetLength() {
		int64_t temp = 1;
		for (const auto& size : Shape) {
			temp *= size;
		}
		return temp;
	}

	void ApplySoftmax() {
		int64_t length = GetLength();
		float MAX = -INFINITY;
		for (int i = 0; i < length; i++) {
			if (*(rawData.get() + i) > MAX) {
				MAX = *(rawData.get() + i);
			}
		}
		std::vector<float> y(length);
		float sum = 0.0f;
		for (int i = 0; i < length; i++) {
			sum += y[i] = std::exp( *(rawData.get() + i) - MAX );
		}
		for (int i = 0; i < length; i++) {
			*(rawData.get() + i) = y[i] / sum;
		}
	}

	//TODO: Make one for GPU
	template <typename T>
	void CreateTensor(T* p_data, int64_t count, bool shouldNormalize = true) {
		int length = GetLength();
		assert(count == length);
		if (shouldNormalize) {
			
			if (rawData == nullptr) {
				rawData = std::make_unique<float[]>(count);
			}
			std::string log;
			for (int i = 0; i < count; i++) {
				log += std::string("   ");
				rawData[i] = 1.0f - ((float)*(p_data + 4*i)) / ((float) std::numeric_limits<T>::max());
				log += std::to_string((int)(rawData[i]));
				if (i % 28 == 0 ) {
					nosEngine.LogI("log: %s", log.c_str());
					log.clear();
				}
			}
			auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
			Value = Ort::Value::CreateTensor<float>(memory_info, rawData.get(), count, Shape.data(), Shape.size());
			return;
		}
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
	}

	//Creates an empty tensor of floats with pre-set shape
	void CreateEmpty() {
		int64_t length = GetLength();

		if (rawData != nullptr)
			rawData.reset();

		rawData = std::make_unique<float[]>(length);
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value = Ort::Value::CreateTensor<float>(memory_info, rawData.get(), length, Shape.data(), Shape.size());
	}

	Ort::Value* GetORTValuePointer() {
		return &Value;
	}

	float* GetData() {
		return rawData.get();
	}

	std::vector<int64_t> GetShape() {
		return Shape;
	}

	int64_t* GetShapePointer() {
		return Shape.data();
	}

private:
	Ort::Value Value{nullptr};
	std::vector<int64_t> Shape;
	std::unique_ptr<float[]> rawData;
};

struct UUIDGenerator {
	UUIDGenerator() = default;

	uuids::uuid_random_generator Generate() {
		std::random_device rd;
		auto seed_data = std::array<int, std::mt19937::state_size>{};
		std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));

		std::seed_seq seed = std::seed_seq(std::begin(seed_data), std::end(seed_data));
		std::mt19937 mtengine(seed);
		uuids::uuid_random_generator generator(mtengine);
		return generator;
	}

	
};