 #pragma once
#include "nosAI/nosAI.h"
#include "nosAICommon.h"

//A smart object for AIModels to handle memory management
class AIModelContainer { //TODO: Inherit from AIModel
public:
	AIModelContainer(ONNXModel* model);
	~AIModelContainer();
private:
	ONNXModel* Model;
};