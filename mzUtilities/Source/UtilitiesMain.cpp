// Copyright MediaZ AS. All Rights Reserved.

// Includes
#include <MediaZ/PluginAPI.h>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "Checkerboard.frag.spv.dat"
#include "Color.frag.spv.dat"
#include "Gradient.frag.spv.dat"
#include "Merge.frag.spv.dat"
#include "Offset.frag.spv.dat"
#include "QuadMerge.frag.spv.dat"
#include "Resize.frag.spv.dat"
#include "SevenSegment.frag.spv.dat"

// Nodes
#include "Merge.hpp"

MZ_INIT();

namespace mz::utilities
{

enum Utilities
{
    Checkerboard,
    Color,
    Gradient,
    Merge,
    Offset,
    QuadMerge,
    Resize,
    SevenSegment,
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
		*outSize = Utilities::Count;
		return MZ_RESULT_SUCCESS;
	}

    for (size_t i = 0; i < Utilities::Count; ++i)
	{
        auto* funcs = &outFunctions[i];
		switch ((Utilities)i)
		{
            case Utilities::Checkerboard:
            {
            	funcs->TypeName = "mz.utilities.Checkerboard";
			    funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
				outSpirvBuf->Data = (void*)(Checkerboard_frag_spv);
				outSpirvBuf->Size = sizeof(Checkerboard_frag_spv);
				return MZ_RESULT_SUCCESS;
			    };
			break;
            }
            case Utilities::Color:
            {
                funcs->TypeName = "mz.utilities.Color";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(Color_frag_spv);
                    outSpirvBuf->Size = sizeof(Color_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Utilities::Gradient:
            {
                funcs->TypeName = "mz.utilities.Gradient";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(Gradient_frag_spv);
                    outSpirvBuf->Size = sizeof(Gradient_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Utilities::Merge:
            {
                RegisterMerge(funcs);
                break;
            }
            case Utilities::Offset:
            {
                funcs->TypeName = "mz.utilities.Offset";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(Offset_frag_spv);
                    outSpirvBuf->Size = sizeof(Offset_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Utilities::QuadMerge:
            {
                funcs->TypeName = "mz.utilities.QuadMerge";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(QuadMerge_frag_spv);
                    outSpirvBuf->Size = sizeof(QuadMerge_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Utilities::Resize:
            {
                // TODO port to new API
                funcs->TypeName = "mz.utilities.Resize";
                break;
            }
            case Utilities::SevenSegment:
            {
                funcs->TypeName = "mz.utilities.SevenSegment";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(SevenSegment_frag_spv);
                    outSpirvBuf->Size = sizeof(SevenSegment_frag_spv);
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