// Copyright Nodos AS. All Rights Reserved.

// Includes
#include <Nodos/PluginAPI.h>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "../Shaders/ColorCorrect.frag.spv.dat"
#include "../Shaders/Diff.frag.spv.dat"
#include "../Shaders/GaussianBlur.frag.spv.dat"
#include "../Shaders/Kuwahara.frag.spv.dat"
#include "../Shaders/KawaseLightStreak.frag.spv.dat"
#include "../Shaders/PremultiplyAlpha.frag.spv.dat"
#include "../Shaders/Sharpen.frag.spv.dat"
#include "../Shaders/Sobel.frag.spv.dat"
#include "../Shaders/Thresholder.frag.spv.dat"
#include "../Shaders/Sampler.frag.spv.dat"

// Nodes
#include "GaussianBlur.hpp"

NOS_INIT();

namespace nos::filters
{

enum Filters : int
{
	ColorCorrect = 0,
	Diff,
	Kuwahara,
	GaussianBlur,
	KawaseLightStreak,
	PremultiplyAlpha,
	Sharpen,
	Sobel,
	Thresholder,
	Sampler,
	Count
};

#define COLOR_PIN_COLOR_IDX 0
#define COLOR_PIN_OUTPUT_IDX 1

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	if (!outList)
	{
		*outSize = Filters::Count;
		return NOS_RESULT_SUCCESS;
	}

#define REGISTER_NODE(NODE) \
	outList[Filters::NODE]->TypeName = NOS_NAME_STATIC("nos.filters." #NODE); \
	outList[Filters::NODE]->GetShaderSource = [](nosShaderSource* src) -> nosResult { \
		src->SpirvBlob.Data = (void*)(NODE##_frag_spv); \
		src->SpirvBlob.Size = sizeof(NODE##_frag_spv); \
        src->GLSLPath = #NODE ".frag"; \
        src->SpirvPath = #NODE ".frag.spv"; \
		return NOS_RESULT_SUCCESS; \
	};
#define REGISTER_NODE_LICENSED(NODE, featureName, featureMessage)								\
		REGISTER_NODE(NODE)																		\
		outList[Filters::NODE]->OnNodeCreated = [](const nosFbNode* node, void** outCtxPtr) {	\
		nosEngine.RegisterFeature(*node->id(), featureName, 1, featureMessage);					\
		};																						\
		outList[Filters::NODE]->OnNodeDeleted = [](void* ctx, nosUUID nodeId) {					\
		nosEngine.UnregisterFeature(nodeId, featureName);										\
		};

	REGISTER_NODE(ColorCorrect);
	REGISTER_NODE(Diff);
	REGISTER_NODE(Kuwahara);
	RegisterGaussianBlur(outList[GaussianBlur]);
	REGISTER_NODE(KawaseLightStreak);
	REGISTER_NODE(PremultiplyAlpha);
	REGISTER_NODE(Sharpen);
	REGISTER_NODE(Sobel);
	REGISTER_NODE(Thresholder);
	REGISTER_NODE(Sampler);
	
	return NOS_RESULT_SUCCESS;
}
}
}
