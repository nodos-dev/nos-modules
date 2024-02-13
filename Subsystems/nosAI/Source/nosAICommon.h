#ifndef AI_COMMON_H_INCLUDED
#define AI_COMMON_H_INCLUDED
#include <onnxruntime_cxx_api.h>
#include "AICommonMacros.h"

struct AIModelContext {
	std::unique_ptr<Ort::SessionOptions> SessionOptions;
	std::unique_ptr<Ort::Session> ModelSession;
	std::unique_ptr<Ort::RunOptions> RunOptions;
	std::vector<Ort::Value> Inputs;
	std::vector<Ort::Value> Outputs;
};

namespace nos::ai 
{
}



#endif //AI_COMMON_H_INCLUDED
