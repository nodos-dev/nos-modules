#include "AIModelContainer.h"

AIModelContainer::AIModelContainer(ONNXModel* model) : Model(model)
{
}

AIModelContainer::AIModelContainer(AIModelContainer&& other)
{
	this->Model = other.Model;
	other.Model = nullptr;
}

AIModelContainer::~AIModelContainer()
{
	if (Model == nullptr)
		return;
	
	delete[] Model->Inputs->Shape.Dimensions;
	for (int i = 0; i < Model->InputCount; i++) {
		delete[] Model->InputNames[i];
	}
	delete[] Model->InputNames;
	delete[] Model->Inputs;
	
	delete[] Model->Outputs->Shape.Dimensions;
	for (int i = 0; i < Model->OutputCount; i++) {
		delete[] Model->OutputNames[i];
	}
	delete[] Model->OutputNames;
	delete[] Model->Outputs;

	delete reinterpret_cast<AIModelContext*>(Model->Model);

	delete Model;
}
