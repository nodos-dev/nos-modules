#pragma once
#include "Nodos/SubsystemAPI.h"
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "nosAICommon.h"

#define CHECK_NOS_RESULT(nosRes) \
	do { \
		nosResult result = nosRes; \
		if (result != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %d.",__FILE__, __LINE__, result); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

class ONNXRunner {
	//Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
public:
	ONNXRunner(ONNXLogLevel logLevel, const char* logID);
	nosResult LoadONNXModel(std::filesystem::path path, ONNXLoadConfig config);

private:
	static void LoggerCallback(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
	std::unique_ptr<Ort::Env> Environment;
	std::unique_ptr<Ort::SessionOptions> SessionOptions;
	std::unique_ptr<Ort::Session> ModelSession;
};