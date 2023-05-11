// Copyright MediaZ AS. All Rights Reserved.

#include "BasicMain.h"
#include "ChannelViewer.frag.spv.dat"
#include <uuid.h>
#include <glm/glm.hpp>

namespace mz
{

struct ChannelViewerContext
{
    mz::fb::UUID NodeID;

    ChannelViewerContext(mz::fb::UUID NodeID) : NodeID(NodeID) {}

    ~ChannelViewerContext()
    {
        DestroyResources();
    }

    void RegisterShaders()
    {
        static bool shadersRegistered = false;
        if (shadersRegistered)
        {
            return;
        }
        GServices.MakeAPICalls(true, app::TRegisterShader{
                                    .key = "ChannelViewer_Shader",
                                    .spirv = ShaderSrc<sizeof(ChannelViewer_frag_spv)>(ChannelViewer_frag_spv) });

        shadersRegistered = true;
    }

    void DestroyResources()
    {
        GServices.MakeAPICalls(true,
            app::TUnregisterPass{
                .key = "ChannelViewer_Pass_" + UUID2STR(NodeID),
            });
    }

    void Run(mz::Args& pins)
    {
        app::TRunPass mergePass;
        mergePass.pass = "ChannelViewer_Pass_" + UUID2STR(NodeID);

        CopyUniformFromPin(mergePass, pins, "Input");

        auto channel = *pins.Get<u32>("Channel");
        auto format  = *pins.Get<u32>("Format");
        
        glm::vec4 val{};
        val[channel&3] = 1;

        constexpr glm::vec3 coeffs[3] = {{.299f, .587f, .114f},{.2126f,.7152f,.0722f},{.2627f, .678f,.0593f}};

        mergePass.inputs.emplace_back(new app::TShaderBinding{
            .var = "Channel",
            .val =  mz::Buffer::From(val)
        });

        mergePass.inputs.emplace_back(new app::TShaderBinding{
            .var = "Format",
            .val =  mz::Buffer::From(glm::vec4(coeffs[format], channel > 3))
        });

        auto out = pins.GetBuffer("Output")->As<mz::fb::TTexture>();
        mergePass.output.reset(&out);
        GServices.MakeAPICall(mergePass, true);
        mergePass.output.release();
    }

};

void RegisterChannelViewer(NodeActionsMap& functions)
{
    auto& actions = functions["mz.ChannelViewer"];
    actions.NodeCreated = [](fb::Node const& node, mz::Args& args, void** ctx) {
        *ctx = new ChannelViewerContext(*node.id());
        auto* channelViewerCtx = (ChannelViewerContext*)*ctx;
        channelViewerCtx->RegisterShaders();
        GServices.MakeAPICalls(true, app::TRegisterPass{ .key = "ChannelViewer_Pass_" + UUID2STR(*node.id()), .shader = "ChannelViewer_Shader" });
    };
    
    actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) {
        delete (ChannelViewerContext*)ctx;
    };

    actions.EntryPoint = [](mz::Args& pins, void* ctx) {
        auto* channelViewerCtx = (ChannelViewerContext*)ctx;
        channelViewerCtx->Run(pins);
        return true;
    };
}

} // namespace mz
