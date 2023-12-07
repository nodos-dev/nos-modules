#pragma once

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