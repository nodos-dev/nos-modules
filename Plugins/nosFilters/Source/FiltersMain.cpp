// Copyright Nodos AS. All Rights Reserved.

// Includes
#include <Nodos/PluginAPI.h>
#include <glm/glm.hpp>

// Nodes
#include "GaussianBlur.hpp"

NOS_INIT();

namespace nos::filters
{

enum Filters : int
{
	GaussianBlur = 0,
	Count
};

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	if (!outList)
	{
		*outSize = Filters::Count;
		return NOS_RESULT_SUCCESS;
	}
	RegisterGaussianBlur(outList[GaussianBlur]);
	return NOS_RESULT_SUCCESS;
}
}
}
