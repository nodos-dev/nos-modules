#include "ONNXRunner.h"

ONNXRunner::ONNXRunner(ONNXLogLevel logLevel, const char* logID)
{
	Environment = std::make_unique<Ort::Env>(static_cast<OrtLoggingLevel>(logLevel), logID, LoggerCallback, this);

}

nosResult ONNXRunner::LoadONNXModel(AIModel* model, std::filesystem::path path, ONNXLoadConfig config)
{
	CHECK_POINTER(model);
	CHECK_MODEL_FORMAT(model, MODEL_FORMAT_ONNX);
	CHECK_PATH(path);
	CHECK_FILE_EXTENSION(path, ".onnx");
	
	//AIModel has dynamically allocated properties which will be destroyed in AIModelContainer's destructor when the ONNXRunner goes out of scope
	//We can not trust users that they will keep the scope of their AIModel and ONNXRunner same, so we are creating our own and then will copy the data
	//of our internalModel to the users model. (i.e. to prevent daggling pointer) 
	AIModel* internalModel = new AIModel;
	memcpy(internalModel, model, sizeof(model));

	ModelContainer.emplace_back(internalModel); //For memory management

	std::unique_ptr<Ort::SessionOptions> Options = std::make_unique<Ort::SessionOptions>();

	switch(config.RunLocation){
		case RUN_ON_CPU:
		{
			if (config.ThreadInfo != nullptr) {
				Options->SetInterOpNumThreads(config.ThreadInfo->InteropThreads);
				Options->SetIntraOpNumThreads(config.ThreadInfo->IntraopThreads);
			}
			if (config.ExecutionMode != nullptr) {
				Options->SetExecutionMode(static_cast<ExecutionMode>(*config.ExecutionMode));
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
			Options->AppendExecutionProvider_CUDA(CUDAOptions);
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
	Options->SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(config.OptimizationLevel));
	Options->SetLogId(config.LogID);
	std::unique_ptr<Ort::Session> ModelSession = std::make_unique<Ort::Session>((*Environment), path.c_str(), Options);
	
	//This will fill the input and output info
	nos::ai::FillAIModelFromSession(internalModel, ModelSession.get());

	AIModelContext* context = new AIModelContext;
	context->SessionOptions = std::move(Options);
	context->ModelSession = std::move(ModelSession);
	internalModel->Model = reinterpret_cast<void*>(context);
	
	//Tranfer updates to user's instance
	memcpy(model, internalModel, sizeof(model));

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
