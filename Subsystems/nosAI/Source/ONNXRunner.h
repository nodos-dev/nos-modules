#ifndef ONNX_RUNNER_H_INCLUDED
#define ONNX_RUNNER_H_INCLUDED
#include "Nodos/SubsystemAPI.h"
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "nosAICommon.h"
#include "nosAI/nosAISubsystem.h"
#include "AIModelContainer.h"


class ONNXRunner {
private:
public:
	ONNXRunner();
	ONNXRunner(ONNXLogLevel logLevel, const char* logID);
	void Initialize(ONNXLogLevel logLevel, const char* logID);
	nosResult LoadONNXModel(ONNXModel* model, std::filesystem::path path, ONNXLoadConfig config);
	nosResult RunONNXModel(ONNXModel* model);
	nosResult SetModelInput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);
	nosResult SetModelOutput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo);

private:
	static void LoggerCallback (void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);
	nosResult FillAIModelFromSession(ONNXModel* model, Ort::Session* session);
	std::unique_ptr<Ort::Env> Environment;

	std::vector<AIModelContainer> ModelContainer; //Crucial for memory management
};

#endif //ONNX_RUNNER_H_INCLUDED