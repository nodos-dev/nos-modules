#include "ONNXRunner.h"
ONNXRunner::ONNXRunner()
{
}

ONNXRunner::ONNXRunner(ONNXLogLevel logLevel, const char* logID)
{
	Environment = std::make_unique<Ort::Env>(static_cast<OrtLoggingLevel>(logLevel), logID, ONNXRunner::LoggerCallback, this);

}

void ONNXRunner::Initialize(ONNXLogLevel logLevel, const char* logID) 
{
	if (!Environment) {
		Environment = std::make_unique<Ort::Env>(static_cast<OrtLoggingLevel>(logLevel), logID, ONNXRunner::LoggerCallback, this);
	}
}

nosResult ONNXRunner::LoadONNXModel(ONNXModel* model, std::filesystem::path path, ONNXLoadConfig config)
{
	CHECK_POINTER(model);
	CHECK_PATH(path);
	CHECK_FILE_EXTENSION(path, ".onnx");
	
	//AIModel has dynamically allocated properties which will be destroyed in AIModelContainer's destructor when the ONNXRunner goes out of scope
	//We can not trust users that they will keep the scope of their AIModel and ONNXRunner same, so we are creating our own and then will copy the data
	//of our internalModel to the users model. (i.e. to prevent daggling pointer) 
	ONNXModel* internalModel = new ONNXModel();
	memcpy(internalModel, model, sizeof(ONNXModel));

	std::unique_ptr<Ort::SessionOptions> Options = std::make_unique<Ort::SessionOptions>();

	switch(config.RunLocation){
		case RUN_ONNX_ON_CPU:
		{
			if (config.ThreadInfo != nullptr) {
				Options->SetInterOpNumThreads(config.ThreadInfo->InteropThreads);
				Options->SetIntraOpNumThreads(config.ThreadInfo->IntraopThreads);
			}
			if (config.ExecutionMode != nullptr) {
				Options->SetExecutionMode(static_cast<ExecutionMode>(*config.ExecutionMode));
			}
			break;
		}
		case RUN_ONNX_ON_CUDA:
		{
			OrtCUDAProviderOptions CUDAOptions = {};
			if (config.CUDAOptions != nullptr) {
				CUDAOptions.device_id = config.CUDAOptions->DeviceID;
				CUDAOptions.has_user_compute_stream = static_cast<int>(config.CUDAOptions->CustomStream);
				CUDAOptions.user_compute_stream = config.CUDAOptions->Stream;
			}
			Options->AppendExecutionProvider_CUDA(CUDAOptions);
			break;
		}
		case RUN_ONNX_ON_TENSORRT:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case RUN_ONNX_ON_DIRECTML:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case RUN_ONNX_ON_ROCM:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
	}
	Options->SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(config.OptimizationLevel));
	//TODO: Create a meta data of the saved ORT model so that we can know if we re-use it or not (if model optimized & saved for CUDA but next time user wants to run it on CPU we cant re-use it)

	/*
		Ort::SessionOptions session_options;
		session_options.AddConfigEntry("session.load_model_format", "ORT");
		
		Ort::Env env;
		Ort::Session session(env, <path to model>, session_options);
	*/
	if (config.OptimizedModelSavePath != nullptr) {
		std::wstring ORT_Path = config.OptimizedModelSavePath; // save the optimized model //TODO: (may be save it to under nodos?)
		Options->SetOptimizedModelFilePath(config.OptimizedModelSavePath);
	}
	Options->SetLogId(config.LogID);
	std::unique_ptr<Ort::Session> ModelSession = std::make_unique<Ort::Session>((*Environment), path.wstring().c_str(), *Options);
	
	//This will fill the input and output info
	FillAIModelFromSession(internalModel, ModelSession.get());

	AIModelContext* context = new AIModelContext;
	context->SessionOptions = std::move(Options);
	context->ModelSession = std::move(ModelSession);
	context->RunOptions = std::make_unique<Ort::RunOptions>();
	internalModel->Model = reinterpret_cast<void*>(context);
	ModelContainer.emplace_back(internalModel); //For memory management

	//Tranfer updates to user's instance
	memcpy(model, internalModel, sizeof(ONNXModel));
	

	return NOS_RESULT_SUCCESS;
}

nosResult ONNXRunner::RunONNXModel(ONNXModel* model)
{
	CHECK_POINTER(model);
	AIModelContext* context = reinterpret_cast<AIModelContext*>(model->Model);
	CHECK_POINTER(context->ModelSession);
	CHECK_POINTER(context->RunOptions);
	context->ModelSession->Run((*context->RunOptions), model->InputNames, context->Inputs.data(), context->Inputs.size(), model->OutputNames, context->Outputs.data(), context->Outputs.size());
	return NOS_RESULT_SUCCESS;
}

nosResult ONNXRunner::SetModelInput(ONNXModel* model, uint32_t inputIndex, void* Data, ParameterMemoryInfo memoryInfo) {
	CHECK_POINTER(model);
	CHECK_INDEX_BOUNDS(inputIndex, model->InputCount);
	AIModelContext* context = reinterpret_cast<AIModelContext*>(model->Model);
	switch (model->Inputs[inputIndex].Type) {
		case TYPE_UNKNOWN:
		{
			CHECK_NOS_RESULT(NOS_RESULT_INVALID_ARGUMENT);
			break;
		}
		case TYPE_TENSOR:
		{
			Ort::MemoryInfo info(nullptr);
			auto allocatorInfo = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, *memoryInfo.CUDADeviceID, OrtMemTypeDefault);
			switch (model->Inputs[inputIndex].ElementType) {
				case ELEMENT_TYPE_UNDEFINED:
					CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
					break;
				case ELEMENT_TYPE_FLOAT:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<float>(allocatorInfo, 
						reinterpret_cast<float*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_UINT8:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<uint8_t>(allocatorInfo,
						reinterpret_cast<uint8_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_INT8:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<int8_t>(allocatorInfo,
						reinterpret_cast<int8_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_UINT16:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<uint16_t>(allocatorInfo,
						reinterpret_cast<uint16_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_INT16:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<int16_t>(allocatorInfo,
						reinterpret_cast<int16_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_INT32:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(allocatorInfo,
						reinterpret_cast<int32_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_INT64:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(allocatorInfo,
						reinterpret_cast<int64_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_STRING:
					CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
					break;
				case ELEMENT_TYPE_BOOL:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<bool>(allocatorInfo,
						reinterpret_cast<bool*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_FLOAT16:
					CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
					break;
				case ELEMENT_TYPE_DOUBLE:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<double>(allocatorInfo,
						reinterpret_cast<double*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_UINT32:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<uint32_t>(allocatorInfo,
						reinterpret_cast<uint32_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_UINT64:
					context->Inputs.emplace_back(Ort::Value::CreateTensor<uint64_t>(allocatorInfo,
						reinterpret_cast<uint64_t*>(Data), memoryInfo.Size, model->Inputs[inputIndex].Shape.Dimensions, model->Inputs[inputIndex].Shape.DimensionCount));
					break;
				case ELEMENT_TYPE_COMPLEX64:
				case ELEMENT_TYPE_COMPLEX128:
				case ELEMENT_TYPE_BFLOAT16:
				case ELEMENT_TYPE_FLOAT8E4M3FN:
				case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
				case ELEMENT_TYPE_FLOAT8E5M2:
				case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
					CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
					break;
			}
			break;
		}  
		case TYPE_SEQUENCE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_MAP:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPAQUE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_SPARSETENSOR:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPTIONAL:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
	}
	return NOS_RESULT_SUCCESS;
}

nosResult ONNXRunner::SetModelOutput(ONNXModel* model, uint32_t outputIndex, void* Data, ParameterMemoryInfo memoryInfo) {
	CHECK_POINTER(model);
	CHECK_INDEX_BOUNDS(outputIndex, model->OutputCount);
	AIModelContext* context = reinterpret_cast<AIModelContext*>(model->Model);
	switch (model->Outputs[outputIndex].Type) {
		case TYPE_UNKNOWN:
		{
			CHECK_NOS_RESULT(NOS_RESULT_INVALID_ARGUMENT);
			break;
		}
		case TYPE_TENSOR:
		{
			auto allocatorInfo = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, *memoryInfo.CUDADeviceID, OrtMemTypeDefault);
			switch (model->Outputs[outputIndex].ElementType) {
			case ELEMENT_TYPE_UNDEFINED:
				CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
				break;
			case ELEMENT_TYPE_FLOAT:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<float>(allocatorInfo,
					reinterpret_cast<float*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_UINT8:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<uint8_t>(allocatorInfo,
					reinterpret_cast<uint8_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_INT8:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<int8_t>(allocatorInfo,
					reinterpret_cast<int8_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_UINT16:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<uint16_t>(allocatorInfo,
					reinterpret_cast<uint16_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_INT16:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<int16_t>(allocatorInfo,
					reinterpret_cast<int16_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_INT32:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<int32_t>(allocatorInfo,
					reinterpret_cast<int32_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_INT64:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<int64_t>(allocatorInfo,
					reinterpret_cast<int64_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_STRING:
				CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
				break;
			case ELEMENT_TYPE_BOOL:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<bool>(allocatorInfo,
					reinterpret_cast<bool*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_FLOAT16:
				CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
				break;
			case ELEMENT_TYPE_DOUBLE:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<double>(allocatorInfo,
					reinterpret_cast<double*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_UINT32:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<uint32_t>(allocatorInfo,
					reinterpret_cast<uint32_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_UINT64:
				context->Outputs.emplace_back(Ort::Value::CreateTensor<uint64_t>(allocatorInfo,
					reinterpret_cast<uint64_t*>(Data), memoryInfo.Size, model->Outputs[outputIndex].Shape.Dimensions, model->Outputs[outputIndex].Shape.DimensionCount));
				break;
			case ELEMENT_TYPE_COMPLEX64:
			case ELEMENT_TYPE_COMPLEX128:
			case ELEMENT_TYPE_BFLOAT16:
			case ELEMENT_TYPE_FLOAT8E4M3FN:
			case ELEMENT_TYPE_FLOAT8E4M3FNUZ:
			case ELEMENT_TYPE_FLOAT8E5M2:
			case ELEMENT_TYPE_FLOAT8E5M2FNUZ:
				CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
				break;
			}
			break;
		}
		case TYPE_SEQUENCE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_MAP:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPAQUE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_SPARSETENSOR:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPTIONAL:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
	}
	return NOS_RESULT_SUCCESS;
}


nosResult ONNXRunner::FillAIModelFromSession(ONNXModel* model, Ort::Session* session) {
	CHECK_POINTER(model);
	CHECK_POINTER(session);
	model->InputCount = session->GetInputCount();
	model->OutputCount = session->GetOutputCount();
	if (model->InputCount > 0) {
		ModelIO* input = new ModelIO[model->InputCount];
		model->Inputs = input;
		model->InputNames = new char* [model->InputCount];
	}
	if (model->OutputCount > 0) {
		ModelIO* output = new ModelIO[model->OutputCount];
		model->Outputs = output;
		model->OutputNames = new char* [model->InputCount];
	}
	for (size_t i = 0; i < model->InputCount; i++) {

		model->Inputs[i].Type = static_cast<IOType>(session->GetInputTypeInfo(i).GetONNXType());
		//Retrieve name information
		Ort::AllocatorWithDefaultOptions ortAllocator = {};
		Ort::AllocatedStringPtr allocatedName = session->GetInputNameAllocated(i, ortAllocator);
		size_t nameLength = strlen(allocatedName.get());
		model->InputNames[i] = new char[nameLength];
		memcpy(model->InputNames[i], allocatedName.get(), nameLength);

		switch (model->Inputs[i].Type)
		{
		case TYPE_UNKNOWN:
		{
			CHECK_NOS_RESULT(NOS_RESULT_INVALID_ARGUMENT);
			break;
		}
		case TYPE_TENSOR:
		{
			//Retrieve tensor information
			auto tensorInfo = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
			auto tensorShape = tensorInfo.GetShape();
			model->Inputs[i].Shape.DimensionCount = tensorShape.size();
			model->Inputs[i].Shape.Dimensions = new int64_t[tensorShape.size()];
			memcpy(model->Inputs[i].Shape.Dimensions, tensorShape.data(), sizeof(int64_t)*tensorShape.size());

			model->Inputs[i].ElementType = static_cast<TensorElementType>(tensorInfo.GetElementType());
			break;
		}
		case TYPE_SEQUENCE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_MAP:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPAQUE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_SPARSETENSOR:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPTIONAL:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		}
	}

	for (size_t i = 0; i < model->OutputCount; i++) {

		model->Outputs[i].Type = static_cast<IOType>(session->GetOutputTypeInfo(i).GetONNXType());
		//Retrieve name information
		Ort::AllocatorWithDefaultOptions ortAllocator = {};
		Ort::AllocatedStringPtr allocatedName = session->GetOutputNameAllocated(i, ortAllocator);
		size_t nameLength = strlen(allocatedName.get());
		model->OutputNames[i] = new char[nameLength];
		memcpy(model->OutputNames[i], allocatedName.get(), nameLength);

		switch (model->Outputs[i].Type)
		{
		case TYPE_UNKNOWN:
		{
			CHECK_NOS_RESULT(NOS_RESULT_INVALID_ARGUMENT);
			break;
		}
		case TYPE_TENSOR:
		{
			//Retrieve tensor information
			auto tensorInfo = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
			auto tensorShape = tensorInfo.GetShape();
			model->Outputs[i].Shape.DimensionCount = tensorShape.size();
			model->Outputs[i].Shape.Dimensions = new int64_t[tensorShape.size()];
			memcpy(model->Outputs[i].Shape.Dimensions, tensorShape.data(), sizeof(int64_t)*tensorShape.size());

			model->Outputs[i].ElementType = static_cast<TensorElementType>(tensorInfo.GetElementType());
			break;
		}
		case TYPE_SEQUENCE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_MAP:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPAQUE:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_SPARSETENSOR:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		case TYPE_OPTIONAL:
		{
			CHECK_NOS_RESULT(NOS_RESULT_NOT_IMPLEMENTED);
			break;
		}
		}
	}
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
