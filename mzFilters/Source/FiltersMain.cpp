// Copyright MediaZ AS. All Rights Reserved.

// Includes
#include <MediaZ/PluginAPI.h>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

// Shaders
#include "Color.frag.spv.dat"
#include "ColorCorrect.frag.spv.dat"
#include "Diff.frag.spv.dat"
#include "Gradient.frag.spv.dat"
#include "GaussianBlur.frag.spv.dat"
#include "Kuwahara.frag.spv.dat"

// Nodes
#include "GaussianBlur.hpp"

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
    GaussianBlur,
    Count
};

#define COLOR_PIN_COLOR_IDX 0
#define COLOR_PIN_OUTPUT_IDX 1

extern "C"
{

MZAPI_ATTR MzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, MzNodeFunctions* outFunctions)
{
    if (!outFunctions) {
        *outSize = Filters::Count;
        return MZ_RESULT_SUCCESS;
    }
    for (size_t i = 0; i < Filters::Count; ++i) {
        auto* funcs = &outFunctions[i];
        switch ((Filters)i) 
        {
            case Filters::Color:
            {
                funcs->TypeName = "mz.Color";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(Color_frag_spv);
                    outSpirvBuf->Size = sizeof(Color_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Filters::ColorCorrect:
            {
                funcs->TypeName = "mz.ColorCorrect";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(ColorCorrect_frag_spv);
                    outSpirvBuf->Size = sizeof(ColorCorrect_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Filters::Diff:
            {
                funcs->TypeName = "mz.Diff";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult {
                    outSpirvBuf->Data = (void*)(Diff_frag_spv);
                    outSpirvBuf->Size = sizeof(Diff_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Filters::Gradient:
            {
                funcs->TypeName = "mz.Gradient";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult
                {
                    outSpirvBuf->Data = (void*)(Gradient_frag_spv);
                    outSpirvBuf->Size = sizeof(Gradient_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Filters::Kuwahara:
            {
                funcs->TypeName = "mz.Kuwahara";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult
                {
                    outSpirvBuf->Data = (void*)(Kuwahara_frag_spv);
                    outSpirvBuf->Size = sizeof(Kuwahara_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                break;
            }
            case Filters::GaussianBlur: {
                funcs->TypeName = "mz.GaussianBlur";
                funcs->GetShaderSource = [](MzBuffer* outSpirvBuf) -> MzResult
                {
                    outSpirvBuf->Data = (void*)(GaussianBlur_frag_spv);
                    outSpirvBuf->Size = sizeof(GaussianBlur_frag_spv);
                    return MZ_RESULT_SUCCESS;
                };
                funcs->ExecuteNode = GaussianBlur_ExecuteNode;
                funcs->OnNodeCreated = GaussianBlur_OnNodeCreated;
                break;
            }
            default:
                return MZ_RESULT_INVALID_ARGUMENT;
        }
    }
    return MZ_RESULT_SUCCESS;
}
}
}