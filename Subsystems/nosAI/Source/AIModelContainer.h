#pragma once
#include "nosAI/nosAI.h"

//A smart object for AIModels to handle memory management
class AIModelContainer {
public:
	AIModelContainer(AIModel* model);
	~AIModelContainer();
private:
	AIModel* Model;
};