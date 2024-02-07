#pragma once
#include "Nodos/SubsystemAPI.h"
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "nosAICommon.h"
#include "nosAI/nosAI.h"
#include "AIModelContainer.h"
   
class ONNXRunner {
public:
	ONNXRunner(ONNXLogLevel logLevel, const char* logID);
	nosResult LoadONNXModel(AIModel* model, std::filesystem::path path, ONNXLoadConfig config);
private:
	static void LoggerCallback (void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
	std::unique_ptr<Ort::Env> Environment;

	std::vector<AIModelContainer> ModelContainer; //Crucial for memory management
};