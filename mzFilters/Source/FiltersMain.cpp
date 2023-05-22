// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/PluginAPI.h>
#include <MediaZ/Helpers.hpp>
#include <Builtins_generated.h>

#include <glm/glm.hpp>

// Shaders
#include "Color.frag.spv.dat"
#include "ColorCorrect.frag.spv.dat"
#include "Diff.frag.spv.dat"
#include "Gradient.frag.spv.dat"

MZ_INIT();

namespace mz::filters
{

enum Filters
{
    Color = 0,
    ColorCorrect,
    Diff,
    Gradient,
    Kuwahara,
    Count
};

#define COLOR_PIN_COLOR_IDX 0
#define COLOR_PIN_OUTPUT_IDX 1

extern "C"
{

MZAPI_ATTR int MZAPI_CALL mzGetNodeTypeCount()
{
    return Filters::Count;
}

MZAPI_ATTR MzResult MZAPI_CALL mzExportNodeFunctions(int nodeTypeIndex, MzNodeFunctions* outFunctions)
{
    switch (nodeTypeIndex){
        case Filters::Color:
        {
            outFunctions->TypeName = "mz.Color";
            outFunctions->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                outSpirvBuf->Data = Color_frag_spv;
                outSpirvBuf->Size = sizeof(Color_frag_spv);
                return MzResult::Success;
            }
        }
        case Filters::ColorCorrect:
        {
            outFunctions->TypeName = "mz.ColorCorrect";
            outFunctions->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                outSpirvBuf->Data = ColorCorrect_frag_spv;
                outSpirvBuf->Size = sizeof(ColorCorrect_frag_spv);
                return MzResult::Success;
            }
        }
        case Filters::Diff:
        {
            outFunctions->TypeName = "mz.Diff";
            outFunctions->GetShaderSource = outFunctions->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                outSpirvBuf->Data = Diff_frag_spv;
                outSpirvBuf->Size = sizeof(Diff_frag_spv);
                return MzResult::Success;
            }
        }
        case Filters::Gradient:
        {
			outFunctions->TypeName = "mz.Gradient";
            outFunctions->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult
            {
				outSpirvBuf->Data = Gradient_frag_spv;
				outSpirvBuf->Size = sizeof(Gradient_frag_spv);
				return MzResult::Success;
			}
		}
        case Filters::Kuwahara:
        {
            outFunctions->TypeName = "mz.Kuwahara";
        }
        default:
            return MzResult::InvalidArgument;
    }
}
}
}