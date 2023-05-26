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

enum Filters
{
	ColorCorrect,					// Done
	Diff,							// Done
	Kuwahara,						// Done
	GaussianBlur,					// Done
	KawaseLightStreak,				// Done
	PremultiplyAlpha,				// Done
	Sharpen,						// Done
	Sobel,							// Done
	Thresholder,					// Done
	Sampler,						// Done
	Count
};

#define COLOR_PIN_COLOR_IDX 0
#define COLOR_PIN_OUTPUT_IDX 1

extern "C"
{

MZAPI_ATTR MzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, MzNodeFunctions* outFunctions)
{
	if (!outFunctions)
	{
		*outSize = Filters::Count;
		return MZ_RESULT_SUCCESS;
	}
	for (size_t i = 0; i < Filters::Count; ++i)
	{
		auto* funcs = &outFunctions[i];
		switch ((Filters)i)
		{
		// COLOR CORRECT FILTER
		case Filters::ColorCorrect: {
			funcs->TypeName = "mz.ColorCorrect";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
				outSpirvBuf->Data = (void*)(ColorCorrect_frag_spv);
				outSpirvBuf->Size = sizeof(ColorCorrect_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// DIFF FILTER
		case Filters::Diff: {
			funcs->TypeName = "mz.Diff";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
				outSpirvBuf->Data = (void*)(Diff_frag_spv);
				outSpirvBuf->Size = sizeof(Diff_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// KUWAHARA FILTER
		case Filters::Kuwahara: {
			funcs->TypeName = "mz.Kuwahara";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
				outSpirvBuf->Data = (void*)(Kuwahara_frag_spv);
				outSpirvBuf->Size = sizeof(Kuwahara_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// GAUSSIAN BLUR FILTER
		case Filters::GaussianBlur: {
			RegisterGaussianBlur(funcs);
			break;
		}
		// MERGE FILTER
		case Filters::KawaseLightStreak:{
			funcs->TypeName = "mz.KawaseLightStreak";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(KawaseLightStreak_frag_spv);
				outSpirvBuf->Size = sizeof(KawaseLightStreak_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// PREMULTIPLY ALPHA FILTER
		case Filters::PremultiplyAlpha: {
			funcs->TypeName = "mz.PremultiplyAlpha";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(PremultiplyAlpha_frag_spv);
				outSpirvBuf->Size = sizeof(PremultiplyAlpha_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// SHARPEN FILTER
		case Filters::Sharpen: {
			funcs->TypeName = "mz.Sharpen";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(Sharpen_frag_spv);
				outSpirvBuf->Size = sizeof(Sharpen_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// SOBEL FILTER
		case Filters::Sobel: {
			funcs->TypeName = "mz.Sobel";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(Sobel_frag_spv);
				outSpirvBuf->Size = sizeof(Sobel_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// THRESHOLDER FILTER
		case Filters::Thresholder: {
			funcs->TypeName = "mz.Thresholder";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(Thresholder_frag_spv);
				outSpirvBuf->Size = sizeof(Thresholder_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		// SAMPLER FILTER
		case Filters::Sampler: {
			funcs->TypeName = "mz.Sampler";
			funcs->GetShaderSource = [](MzBuffer* outSpirvBuf)-> MzResult {
				outSpirvBuf->Data = (void*)(Sampler_frag_spv);
				outSpirvBuf->Size = sizeof(Sampler_frag_spv);
				return MZ_RESULT_SUCCESS;
			};
			break;
		}
		default: break;
		}
	}
	return MZ_RESULT_SUCCESS;
}
}
}
