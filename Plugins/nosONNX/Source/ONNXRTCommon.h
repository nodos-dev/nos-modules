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

	void SetShape(std::vector<int64_t>&& shape ) {
		Shape = std::move(shape);
	}

	void SetShape(std::vector<int32_t> shape) {
		std::copy(shape.begin(), shape.end(), Shape.begin());
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

	//TODO: Make one for GPU
	template <typename T>
	void SetData(T* p_data, int32_t count, bool isNormalized = false) {
		if (!isNormalized) {
			for (int i = 0; i < count; i++) {
				*(p_data + i) = *(p_data + i) / std::numeric_limits<T>::max();
			}
		}
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
	}

	Ort::Value* GetData() {
		return &Value;
	}

	std::vector<int64_t> GetShape() {
		return Shape;
	}

private:
	Ort::Value Value{nullptr};
	std::vector<int64_t> Shape;
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