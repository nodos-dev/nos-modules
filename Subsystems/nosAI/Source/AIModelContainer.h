#ifndef AI_MODEL_CONTAINER_H_INCLUDED
#define AI_MODEL_CONTAINER_H_INCLUDED
#include "Nodos/SubsystemAPI.h" 
#include "nosAI/nosAISubsystem.h"
#include "nosAICommon.h"

//A smart object for AIModels to handle memory management
class AIModelContainer { //TODO: Inherit from AIModel
public:
	AIModelContainer(ONNXModel* model);
	AIModelContainer(AIModelContainer&& other);
	AIModelContainer(const AIModelContainer& other) = delete;
	~AIModelContainer();
private:
	ONNXModel* Model;
};
#endif //AI_MODEL_CONTAINER_H_INCLUDED