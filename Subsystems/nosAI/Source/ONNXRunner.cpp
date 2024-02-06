#include "ONNXRunner.h"

ONNXRunner::ONNXRunner(ONNXLogLevel logLevel, const char* logID)
{
	Environment = std::make_unique<Ort::Env>(static_cast<OrtLoggingLevel>(logLevel), logID, LoggerCallback, this);

}

nosResult ONNXRunner::LoadONNXModel(std::filesystem::path path, ONNXLoadConfig config)
{
	Ort::SessionOptions Options;
	switch(config.RunLocation){
		case RUN_ON_CPU:
		{
			if (config.ThreadInfo != nullptr) {
				Options.SetInterOpNumThreads(config.ThreadInfo->InteropThreads);
				Options.SetIntraOpNumThreads(config.ThreadInfo->IntraopThreads);
			}
			if (config.ExecutionMode != nullptr) {
				Options.SetExecutionMode(static_cast<ExecutionMode>(*config.ExecutionMode));
			}

		}
		case RUN_ON_CUDA:
		{
			OrtCUDAProviderOptions CUDAOptions = {};
			if (config.CUDAOptions != nullptr) {
				CUDAOptions.device_id = config.CUDAOptions->DeviceID;
				CUDAOptions.has_user_compute_stream = static_cast<int>(config.CUDAOptions->CustomStream);
				CUDAOptions.user_compute_stream = config.CUDAOptions->Stream;
			}
			Options.AppendExecutionProvider_CUDA(CUDAOptions);
		}
		case RUN_ON_TENSORRT:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}
		case RUN_ON_DIRECTML:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}
		case RUN_ON_ROCM:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
		}
	}
	Options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(config.OptimizationLevel));

	ModelSession = std::make_unique<Ort::Session>((*Environment), path.c_str(), Options);
	return NOS_RESULT_SUCCESS;
}

void ONNXRunner::LoggerCallback(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message)
{
	ONNXRunner* currentContext = reinterpret_cast<ONNXRunner*>(param);
	switch (severity) {
		case ORT_LOGGING_LEVEL_ERROR:
		case ORT_LOGGING_LEVEL_FATAL: nosEngine.LogE("ONNXRT: [%s]: %s", logid, message); break;
		case ORT_LOGGING_LEVEL_INFO: nosEngine.LogI("ONNXRT: [%s]: %s", logid, message); break;
		case ORT_LOGGING_LEVEL_WARNING: nosEngine.LogW("ONNXRT: [%s]: %s", logid, message); break;
		default: break;
	}
}
