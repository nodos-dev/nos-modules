// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"

// Shaders
#include "GaussianBlur.frag.spv.dat"
#include "KawaseLightStreak.frag.spv.dat"
#include "Kuwahara.frag.spv.dat"
#include "Offset.frag.spv.dat"
#include "QuadMerge.frag.spv.dat"
#include "Sampler.frag.spv.dat"
#include "Sharpen.frag.spv.dat"
#include "Sobel.frag.spv.dat"
#include "Thresholder.frag.spv.dat"
#include "PremultiplyAlpha.frag.spv.dat"
#include "Gradient.frag.spv.dat"
#include "Checkerboard.frag.spv.dat"
#include "SevenSegment.frag.spv.dat"
#include "Diff.frag.spv.dat"
#include "ColorCorrect.frag.spv.dat"
#include "Color.frag.spv.dat"

// std
#include <filesystem>
#include <fstream>

namespace mz
{

void RegisterGaussianBlur(NodeActionsMap& functions);
void RegisterResize(NodeActionsMap& functions);
void RegisterMerge(NodeActionsMap& functions);

void RegisterFilters(NodeActionsMap& functions)
{
    auto ShaderReloader = [](std::string const &className, std::string const &path) {
        auto tmp = std::filesystem::temp_directory_path();
        auto stem = std::filesystem::path(path).stem().string();
        std::string out = tmp.string() + "/" + stem + ".frag";
        std::string cmd = "glslc " + path + " -c -o " + out;

        if (system(cmd.c_str()))
        {
            return;
        }
        auto spirv = ReadSpirv(out.c_str());
        std::string shaderName = "$$GPUJOBSHADER$$" + className;
        std::string passName = "$$GPUJOBPASS$$" + className;
        GServices.MakeAPICalls(true, app::TRegisterShader{.key = shaderName, .spirv = spirv},
                              app::TRegisterPass{.key = passName, .shader = shaderName});
    };

#define REGISTER_NODE(NODE) \
    functions["mz."#NODE] = { \
        .ShaderSource = [] { return ShaderSrc<sizeof(NODE##_frag_spv)>(NODE##_frag_spv); }, \
        .NodeFunctions = {{"ReloadShaders", [ShaderReloader](auto&, auto&, auto id) { ShaderReloader("mz."#NODE, MZ_REPO_ROOT "/Plugins/mzBasic/Source/Filters/" #NODE ".frag"); }}} \
    };
    
    REGISTER_NODE(KawaseLightStreak);
    REGISTER_NODE(Thresholder);
    REGISTER_NODE(Sampler);
    REGISTER_NODE(Offset);
    REGISTER_NODE(QuadMerge);
    REGISTER_NODE(Sobel);
    REGISTER_NODE(Kuwahara);
    REGISTER_NODE(Sharpen);
    REGISTER_NODE(PremultiplyAlpha);
    REGISTER_NODE(Gradient);
    REGISTER_NODE(Checkerboard);
    REGISTER_NODE(SevenSegment);

    RegisterGaussianBlur(functions);
    RegisterResize(functions);
    RegisterMerge(functions);

    functions["mz.ColorCorrect"] = NodeActions{ .ShaderSource = [] { return ShaderSrc<sizeof(ColorCorrect_frag_spv)>(ColorCorrect_frag_spv); } };
    functions["mz.Diff"] = NodeActions{ .ShaderSource = [] { return ShaderSrc<sizeof(Diff_frag_spv)>(Diff_frag_spv); } };
    functions["mz.Color"] = NodeActions{ .ShaderSource = [] { return ShaderSrc<sizeof(Color_frag_spv)>(Color_frag_spv); } };
}

} // namespace mz