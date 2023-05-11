// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"

#include "Resize.frag.spv.dat"

namespace mz
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
    std::string shaderName = "Shader_mz.Resize";
    std::string passName = "Pass_mz.Resize";
    GServices.MakeAPICalls(true, app::TRegisterShader{.key = shaderName, .spirv = spirv},
                            app::TRegisterPass{.key = passName, .shader = shaderName});
};

void RegisterResize(NodeActionsMap& functions)
{
    functions["mz.Resize"] = NodeActions{
        .NodeCreated = [] (fb::Node const& node, auto& pins, auto ctx) 
        { 
            *ctx = node.UnPack(); 
            static bool reg = false;
            if(reg) return;
            reg = true;
            GServices.MakeAPICalls(true, 
                app::TRegisterShader { .key = "Shader_mz.Resize", .spirv = ShaderSrc<sizeof(Resize_frag_spv)>(Resize_frag_spv) }, 
                app::TRegisterPass { .key = "Pass_mz.Resize", .shader = "Shader_mz.Resize" });
        },
        .NodeUpdate = [] (fb::Node const& node, auto ctx) { node.UnPackTo((fb::TNode*)ctx); },
        .ShaderSource = [] { return ShaderSrc<sizeof(Resize_frag_spv)>(Resize_frag_spv); },
        .NodeRemoved = [] (void* ctx, mz::fb::UUID const& id) { delete (fb::TNode*)ctx; },
        .EntryPoint = [](mz::Args& args, auto ctx) 
        { 
            auto Node = (fb::TNode*)ctx;
            auto Size = *args.GetBuffer("Size")->As<fb::vec2u>();
            auto Out  = args.GetBuffer("Output")->As<fb::TTexture>();

            if(fb::vec2u(Out.width, Out.height) != Size)
            {
                Out.width  = Size.x();
                Out.height = Size.y();
                Out.unscaled = true;
                GServices.Destroy(Out);
                GServices.Create(Out);
                *args.GetBuffer("Output") = mz::Buffer::From(Out);
            }
            
            app::TRunPass2 pass;
            pass.pass = "Pass_mz.Resize",
            pass.output.reset(new fb::TTexture(Out));
            pass.draws.emplace_back(new app::TDrawCall{});
            pass.draws.back()->inputs.emplace_back(new app::TShaderBinding { .var = "Method", .val = *args.GetBuffer("Method") });
            pass.draws.back()->inputs.emplace_back(new app::TShaderBinding { .var = "Input" , .val = *args.GetBuffer("Input")});
            GServices.MakeAPICall(pass, true);
            return true; 
        },
        .NodeFunctions = {{"ReloadShaders" , [](auto&, auto&, auto) { ShaderReloader("mz.Resize", MZ_REPO_ROOT "/Plugins/mzBasic/Source/Filters/Resize.frag"); }}}
    };
}

}