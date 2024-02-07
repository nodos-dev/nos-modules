#include "AIModelContainer.h"

AIModelContainer::AIModelContainer(AIModel* model) : Model(model)
{
}

AIModelContainer::~AIModelContainer()
{
	if (Model == nullptr)
		return;
	
	delete[] Model->Inputs->Shape.Dimensions;
	delete[] Model->Inputs->Name;
	delete[] Model->Inputs;
	
	delete[] Model->Outputs->Shape.Dimensions;
	delete[] Model->Outputs->Name;
	delete[] Model->Outputs;

	delete reinterpret_cast<nos::ai::AIModelContext*>(Model->Model);

	delete Model;
}
