#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>

struct nosTensor
{
public:
	nosTensor() = default;
	nosTensor(size_t channel, size_t width, size_t height) {
		Shape = std::array<size_t, 4>{1, channel, width, height};
	}

	void SetShape(size_t channel, size_t width, size_t height) {
		Shape = std::array<size_t, 4>{1, channel, width, height};
	}

	void SetShape(std::array<size_t, 4>&& shape) { 
		Shape = std::move(shape);
	}

	void SetShape(std::vector<int64_t> shape) { 
		
		for (int i = 0; i < shape.size() && (0 < (4 - shape.size()) ); i++) {
			shape.insert(shape.begin(), 1);
		}

		std::copy_n(shape.begin(), 4, Shape.begin());
	}

	void SetData(Ort::Value&& data) { 
		Value = std::move(data);
	}

	//TODO: Make one for GPU
	template <typename T>
	void SetData(T* p_data, size_t count) { 
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value = Ort::Value::CreateTensor<T>(memory_info, p_data, count, Shape.data(), Shape.size());
	}

	Ort::Value* GetData() {
		return &Value;
	}

private:
	Ort::Value Value{nullptr};
	std::array<size_t, 4> Shape;
};

namespace nos
{
	std::seed_seq Seed()
	{
		std::random_device rd;
		auto seed_data = std::array<int, std::mt19937::state_size>{};
		std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
		return std::seed_seq(std::begin(seed_data), std::end(seed_data));
	}

	std::seed_seq seed = Seed();
	std::mt19937 mtengine(seed);
	uuids::uuid_random_generator generator(mtengine);

}