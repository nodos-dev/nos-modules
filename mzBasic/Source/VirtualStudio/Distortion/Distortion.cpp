// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"

#include "Distort.frag.spv.dat"
#include "Undistort.frag.spv.dat"

namespace mz
{

void RegisterDistortion(NodeActionsMap& functions)
{
    auto ShaderReloader = [](std::string const &nodeId, std::string const &path) {
        auto tmp = std::filesystem::temp_directory_path();
        auto stem = std::filesystem::path(path).stem().string();
        std::string out = tmp.string() + "/" + stem + ".frag";
        std::string cmd = "glslc " + path + " -c -o " + out;

        if (system(cmd.c_str()))
        {
            return;
        }
        auto spirv = ReadSpirv(out.c_str());
        std::string shaderName = "$$GPUJOBSHADER$$" + nodeId;
        std::string passName = "$$GPUJOBPASS$$" + nodeId;
        GServices.MakeAPICalls(true, app::TRegisterShader{.key = shaderName, .spirv = spirv},
                              app::TRegisterPass{.key = passName, .shader = shaderName});
    };

#define REGISTER_NODE(NODE) \
    functions["mz."#NODE] = { \
        .ShaderSource = [] { return ShaderSrc<sizeof(NODE##_frag_spv)>(NODE##_frag_spv); }, \
        .NodeFunctions = {{"ReloadShaders", [ShaderReloader](auto&, auto&, auto id) { ShaderReloader(UUID2STR(*(mz::fb::UUID*)id), MZ_REPO_ROOT "/Plugins/mzBasic/Source/VirtualStudio/Distortion/" #NODE ".frag"); }}} \
    };

    REGISTER_NODE(Undistort);

    functions["mz.Distort"] = NodeActions{
        .ShaderSource = [] { return ShaderSrc<sizeof(Distort_frag_spv)>(Distort_frag_spv); }
    };
}

}