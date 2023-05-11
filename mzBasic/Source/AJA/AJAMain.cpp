// Copyright MediaZ AS. All Rights Reserved.

#include "AJAMain.h"
#include "CopyThread.h"
#include "Ring.h"
#include "AJAClient.h"

#include "RGB2YCbCr.comp.spv.dat"
#include "RGB2YCbCr.frag.spv.dat"
#include "YCbCr2RGB.comp.spv.dat"
#include "YCbCr2RGB.frag.spv.dat"

using namespace mz;

namespace mz
{

static mz::fb::String256 Str256(std::string const &str)
{
    mz::fb::String256 re = {};
    memcpy(re.mutable_val()->data(), str.data(), str.size());
    return re;
}

void mzPluginSDK_API RegisterAJA(NodeActionsMap& functions)
{
    auto &actions = functions["AJA.AJAIn"];

    actions.Shaders = []() {
        return ShaderLibrary {
                {"AJA_YCbCr2RGB_Shader", ShaderSrc<sizeof(YCbCr2RGB_frag_spv)>(YCbCr2RGB_frag_spv)},
                {"AJA_RGB2YCbCr_Shader", ShaderSrc<sizeof(RGB2YCbCr_frag_spv)>(RGB2YCbCr_frag_spv)},
                {"AJA_RGB2YCbCr_Compute_Shader", ShaderSrc<sizeof(RGB2YCbCr_comp_spv)>(RGB2YCbCr_comp_spv)},
                {"AJA_YCbCr2RGB_Compute_Shader", ShaderSrc<sizeof(YCbCr2RGB_comp_spv)>(YCbCr2RGB_comp_spv)},
        };
    };

    actions.Passes = []() {
        return PassLibrary {
            app::TRegisterPass{.key = "AJA_RGB2YCbCr_Compute_Pass", .shader = "AJA_RGB2YCbCr_Compute_Shader"},
            app::TRegisterPass{.key = "AJA_YCbCr2RGB_Compute_Pass", .shader = "AJA_YCbCr2RGB_Compute_Shader"},
            app::TRegisterPass{.key = "AJA_YCbCr2RGB_Pass", .shader = "AJA_YCbCr2RGB_Shader"},
            app::TRegisterPass{.key = "AJA_RGB2YCbCr_Pass", .shader = "AJA_RGB2YCbCr_Shader"}
        };
    };

    actions.CanCreate = [](fb::Node const &node) {
        for (auto pin : *node.pins())
        {
            if (pin->name()->str() == "Device")
            {
                if (flatbuffers::IsFieldPresent(pin, mz::fb::Pin::VT_DATA))
                    return AJADevice::DeviceAvailable((char *)pin->data()->Data(),
                                                      node.class_name()->str() == "AJA.AJAIn");
                break;
            }
        }
        return AJADevice::GetAvailableDevice(node.class_name()->str() == "AJA.AJAIn");
    };

    actions.NodeCreated = [](fb::Node const &node, Args &args, void **ctx) {
        AJADevice::Init();
        const bool isIn = node.class_name()->str() == "AJA.AJAIn";
        AJADevice *dev = 0;

        if (auto devpin = std::find_if(node.pins()->begin(), node.pins()->end(),
                                       [](auto pin) {
                                           return pin->name()->str() == "Device" &&
                                                  flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA);
                                       });
            devpin != node.pins()->end())
        {
            dev = AJADevice::GetDevice((char *)devpin->data()->Data()).get();
        }
        else
            AJADevice::GetAvailableDevice(isIn, &dev);

        if (!dev)
        {
            // This case shouldn't happen with canCreate
            AJADevice::Deinit();
            return;
        }

        for (auto dev : AJADevice::Devices)
        {
            const auto str256 = Str256(dev->GetDisplayName() + "-AJAOut-Reference-Source");
            flatbuffers::FlatBufferBuilder fbb;
            std::vector<mz::fb::String256> list = {Str256("Reference In"), Str256("Free Run")};

            for (int i = 1; i <= NTV2DeviceGetNumVideoInputs(dev->ID); ++i)
            {
                list.push_back(Str256("SDI In " + std::to_string(i)));
            }

            GServices.HandleEvent(CreateAppEvent(
                fbb, mz::app::CreateUpdateStringList(fbb, mz::fb::CreateString256ListDirect(fbb, &str256, &list))));
        }

        AJAClient *c = new AJAClient(isIn, dev);
        *ctx = c;

        std::string refSrc = NTV2ReferenceSourceToString(c->Ref, true);
        flatbuffers::FlatBufferBuilder fbb;

        std::vector<flatbuffers::Offset<mz::fb::Pin>> pinsToAdd;
        using mz::fb::ShowAs;
        using mz::fb::CanShowAs;

        PinMapping mapping;
        auto loadedPins = mapping.Load(node);

        AddIfNotFound("Device", "string", StringValue(dev->GetDisplayName()), loadedPins, pinsToAdd, fbb,
                      ShowAs::PROPERTY, CanShowAs::PROPERTY_ONLY);
        if (auto val = AddIfNotFound("Dispatch Size", "mz.fb.vec2u", mz::Buffer::From(mz::fb::vec2u(c->DispatchSizeX, c->DispatchSizeY)),
                                     loadedPins, pinsToAdd, fbb))
        {
            c->DispatchSizeX = ((glm::uvec2 *)val)->x;
            c->DispatchSizeY = ((glm::uvec2 *)val)->y;
        }

        if (auto val = AddIfNotFound("Shader Type", "AJA.Shader", mz::Buffer::From(ShaderType(c->Shader)), loadedPins,
                                     pinsToAdd, fbb))
        {
            c->Shader = *((ShaderType *)val);
        }

        if (auto val = AddIfNotFound("Debug", "uint", mz::Buffer::From(u32(c->Debug)), loadedPins, pinsToAdd, fbb))
        {
            c->Debug = *((u32 *)val);
        }

        if (auto ref = loadedPins["ReferenceSource"])
        {
            if (flatbuffers::IsFieldPresent(ref, fb::Pin::VT_DATA))
            {
                refSrc = (char *)ref->data()->Data();
            }
        }
        else if (!isIn)
        {
            std::vector<u8> data = StringValue(refSrc);
            pinsToAdd.push_back(mz::fb::CreatePinDirect(
                fbb, generator(), "ReferenceSource", "string", mz::fb::ShowAs::PROPERTY,
                mz::fb::CanShowAs::PROPERTY_ONLY, 0,
                mz::fb::CreateVisualizerDirect(fbb, mz::fb::VisualizerType::COMBO_BOX,
                                               (dev->GetDisplayName() + "-AJAOut-Reference-Source").c_str()),
                &data));
        }
        c->SetReference(refSrc);

        std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg;
        std::vector<mz::fb::UUID> pinsToDel;
        c->OnNodeUpdate(std::move(mapping), loadedPins, pinsToDel);
        c->UpdateStatus(fbb, msg);
        GServices.HandleEvent(
            CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->Mapping.NodeId, ClearFlags::NONE, &pinsToDel,
                                                                  &pinsToAdd, 0, 0, 0, 0, &msg)));
    };

    actions.NodeUpdate = [](auto &node, auto ctx) { ((AJAClient *)ctx)->OnNodeUpdate(node); };
    actions.MenuFired = [](auto &node, auto ctx, auto &request) { ((AJAClient *)ctx)->OnMenuFired(node, request); };
    actions.CommandFired = [](auto &node, auto ctx, u32 cmd) { ((AJAClient *)ctx)->OnCommandFired(cmd); };
    actions.NodeRemoved = [](auto ctx, auto &id) {
        auto c = ((AJAClient *)ctx);
        c->OnNodeRemoved();
        delete c;
    };
    actions.PinValueChanged = [](auto ctx, auto &id, mz::Buffer *value) {
        return ((AJAClient *)ctx)->OnPinValueChanged(id, value->data());
    };
    actions.PinShowAsChanged = [](auto ctx, auto &id, int value) {
        return ((AJAClient *)ctx)->OnPinShowAsChanged(id, value);
    };
    actions.BeginCopyTo = [](auto ctx, CopyInfo &cpy) { return ((AJAClient *)ctx)->BeginCopyTo(cpy); };
    actions.BeginCopyFrom = [](auto ctx, CopyInfo &cpy) { return ((AJAClient *)ctx)->BeginCopyFrom(cpy); };
    actions.EndCopyFrom = [](auto ctx, CopyInfo &cpy) { ((AJAClient *)ctx)->EndCopyFrom(cpy); };
    actions.EndCopyTo = [](auto ctx, CopyInfo &cpy) { ((AJAClient *)ctx)->EndCopyTo(cpy); };
    actions.EntryPoint = [](mz::Args &pins, auto ctx) { return true; };
    actions.OnPathCommand = [](fb::UUID pinID, app::PathCommand command, mz::Buffer *params, void *ctx) {
        auto aja = ((AJAClient *)ctx);
        aja->OnPathCommand(pinID, command, params);
    };
    actions.NodeFunctions["DumpInfo"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
        auto aja = ((AJAClient *)ctx);

        for (u32 i = 0; i < 4; ++i)
        {
            AJALabelValuePairs info = {};
            aja->Device->GetVPID(NTV2Channel(i)).GetInfo(info);
            std::ostringstream ss;
            for (auto &[k, v] : info)
            {
                ss << k << " : " << v << "\n";
            }

            GServices.Log(aja->Device->GetDisplayName() + " SingleLink " + std::to_string(i + 1) + " info", ss.str());
        }
    };

    actions.NodeFunctions["StartLog"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {

    };

    actions.NodeFunctions["StopLog"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {

    };

    actions.NodeFunctions["ReloadShaders"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
        system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/YCbCr2RGB.frag -c -o " MZ_REPO_ROOT
               "/../YCbCr2RGB_.frag");
        system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/RGB2YCbCr.frag -c -o " MZ_REPO_ROOT
               "/../RGB2YCbCr_.frag");
        system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/RGB2YCbCr.comp -c -o " MZ_REPO_ROOT
               "/../RGB2YCbCr_.comp");
        system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/YCbCr2RGB.comp -c -o " MZ_REPO_ROOT
               "/../YCbCr2RGB_.comp");

        system("spirv-opt -O " MZ_REPO_ROOT "/../YCbCr2RGB_.frag -o " MZ_REPO_ROOT "/../YCbCr2RGB.frag");
        system("spirv-opt -O " MZ_REPO_ROOT "/../RGB2YCbCr_.frag -o " MZ_REPO_ROOT "/../RGB2YCbCr.frag");
        system("spirv-opt -O " MZ_REPO_ROOT "/../RGB2YCbCr_.comp -o " MZ_REPO_ROOT "/../RGB2YCbCr.comp");
        system("spirv-opt -O " MZ_REPO_ROOT "/../YCbCr2RGB_.comp -o " MZ_REPO_ROOT "/../YCbCr2RGB.comp");
        auto YCbCr2RGB = ReadSpirv(MZ_REPO_ROOT "/../YCbCr2RGB.frag");
        auto RGB2YCbCr = ReadSpirv(MZ_REPO_ROOT "/../RGB2YCbCr.frag");
        auto RGB2YCbCr2 = ReadSpirv(MZ_REPO_ROOT "/../RGB2YCbCr.comp");
        auto YCbCr2RGB2 = ReadSpirv(MZ_REPO_ROOT "/../YCbCr2RGB.comp");

        for (auto c : AJAClient::Ctx.Clients)
            for (auto& p : c->Pins)
                p->Stop();

        GServices.MakeAPICalls(
            true, app::TRegisterShader{.key = "AJA_YCbCr2RGB_Shader", .spirv = YCbCr2RGB},
            app::TRegisterShader{.key = "AJA_RGB2YCbCr_Shader", .spirv = RGB2YCbCr},
            app::TRegisterShader{.key = "AJA_RGB2YCbCr_Compute_Shader", .spirv = RGB2YCbCr2},
            app::TRegisterShader{.key = "AJA_YCbCr2RGB_Compute_Shader", .spirv = YCbCr2RGB2},
            app::TRegisterPass{.key = "AJA_RGB2YCbCr_Compute_Pass", .shader = "AJA_RGB2YCbCr_Compute_Shader"},
            app::TRegisterPass{.key = "AJA_YCbCr2RGB_Compute_Pass", .shader = "AJA_YCbCr2RGB_Compute_Shader"},
            app::TRegisterPass{.key = "AJA_YCbCr2RGB_Pass", .shader = "AJA_YCbCr2RGB_Shader"},
            app::TRegisterPass{.key = "AJA_RGB2YCbCr_Pass", .shader = "AJA_RGB2YCbCr_Shader"});
        
        for (auto c : AJAClient::Ctx.Clients)
            for (auto& p : c->Pins)
                p->StartThread();
    };

    functions["AJA.AJAOut"] = actions;
}

} // namespace mz
