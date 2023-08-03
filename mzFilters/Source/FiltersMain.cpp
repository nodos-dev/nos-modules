// Copyright MediaZ AS. All Rights Reserved.

// Includes
#include <MediaZ/PluginAPI.h>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "ColorCorrect.frag.spv.dat"
#include "Diff.frag.spv.dat"
#include "GaussianBlur.frag.spv.dat"
#include "Kuwahara.frag.spv.dat"
#include "KawaseLightStreak.frag.spv.dat"
#include "PremultiplyAlpha.frag.spv.dat"
#include "Sharpen.frag.spv.dat"
#include "Sobel.frag.spv.dat"
#include "Thresholder.frag.spv.dat"
#include "Sampler.frag.spv.dat"

// Nodes
#include "GaussianBlur.hpp"

MZ_INIT();

namespace mz::filters
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

MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions** outList)
{
	if (!outList)
	{
		*outSize = Filters::Count;
		return MZ_RESULT_SUCCESS;
	}

#define REGISTER_NODE(NODE) \
	outList[Filters::##NODE]->TypeName = MZ_NAME_STATIC("mz.filters." #NODE); \
	outList[Filters::##NODE]->GetShaderSource = [](mzShaderSource* src) -> mzResult { \
		src->SpirvBlob.Data = (void*)(NODE##_frag_spv); \
		src->SpirvBlob.Size = sizeof(NODE##_frag_spv); \
		return MZ_RESULT_SUCCESS; \
	};

	REGISTER_NODE(ColorCorrect);
	REGISTER_NODE(Diff);
	REGISTER_NODE(Kuwahara);
	REGISTER_NODE(GaussianBlur);
	REGISTER_NODE(KawaseLightStreak);
	REGISTER_NODE(PremultiplyAlpha);
	REGISTER_NODE(Sharpen);
	REGISTER_NODE(Sobel);
	REGISTER_NODE(Thresholder);
	REGISTER_NODE(Sampler);
	
	return MZ_RESULT_SUCCESS;
}
}
}
