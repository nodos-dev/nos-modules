#pragma once
#ifndef AI_COMMON_H_INCLUDED
#define AI_COMMON_H_INCLUDED
#include "nosAI/nosAI.h"
#include <onnxruntime_cxx_api.h>
#include "AICommonMacros.h"

namespace nos::ai {
	static nosResult FillAIModelFromSession(AIModel* model, Ort::Session* session) {
		CHECK_POINTER(model);
		CHECK_POINTER(session);
		model->InputCount = session->GetInputCount();
		model->OutputCount = session->GetOutputCount();

		if (model->InputCount > 0) {
			ModelIO* input = new ModelIO[model->InputCount];
			model->Inputs = input;
		}
		if (model->OutputCount > 0) {
			ModelIO* output = new ModelIO[model->OutputCount];
			model->Outputs = output;
		}
		for (size_t i = 0; i < model->InputCount; i++) {

			model->Inputs[i].Type = static_cast<IOType>(session->GetInputTypeInfo(i).GetONNXType());
			//Retrieve name information
			Ort::AllocatorWithDefaultOptions ortAllocator = {};
			Ort::AllocatedStringPtr allocatedName = session->GetInputNameAllocated(i, ortAllocator);
			size_t nameLength = strlen(allocatedName.get());
			model->Inputs[i].Name = new char[nameLength];
			memcpy(model->Inputs[i].Name, allocatedName.get(), nameLength);
			
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
					memcpy(model->Inputs[i].Shape.Dimensions, tensorShape.data(), tensorShape.size());

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
			model->Outputs[i].Name = new char[nameLength];
			memcpy(model->Outputs[i].Name, allocatedName.get(), nameLength);

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
					memcpy(model->Outputs[i].Shape.Dimensions, tensorShape.data(), tensorShape.size());

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
}



#endif //AI_COMMON_H_INCLUDED
