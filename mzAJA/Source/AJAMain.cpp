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

MZ_INIT();

namespace mz
{

static MzBuffer Blob2Buf(std::vector<u8> const& v) 
{ 
    return { (void*)v.data(), v.size() }; 
};

static mz::fb::String256 Str256(std::string const &str)
{
    mz::fb::String256 re = {};
    memcpy(re.mutable_val()->data(), str.data(), str.size());
    return re;
}

struct AJA
{
    static MzResult GetShaders(size_t* outCount, MzBuffer* outSpirvBufs)
    {
        MzBuffer shaders[] = 
        {
            {(void*)YCbCr2RGB_frag_spv, sizeof(YCbCr2RGB_frag_spv) & ~3},
            {(void*)RGB2YCbCr_frag_spv, sizeof(RGB2YCbCr_frag_spv) & ~3},
            {(void*)RGB2YCbCr_comp_spv, sizeof(RGB2YCbCr_comp_spv) & ~3},
            {(void*)YCbCr2RGB_comp_spv, sizeof(YCbCr2RGB_comp_spv) & ~3},
        };

        *outCount = sizeof(shaders)/sizeof(shaders[0]);
        if(!outSpirvBufs) 
            return MZ_RESULT_SUCCESS;

        for(auto s : shaders)
            *outSpirvBufs++ = s;

        return MZ_RESULT_SUCCESS;
    };

    static MzResult GetPasses(size_t* outCount, MzPassInfo* outMzPassInfos)
    {
        MzPassInfo passes[] = 
        {
            {.Key = "AJA_RGB2YCbCr_Compute_Pass", .Shader = "AJA_RGB2YCbCr_Compute_Shader", .MultiSample = 1},
            {.Key = "AJA_YCbCr2RGB_Compute_Pass", .Shader = "AJA_YCbCr2RGB_Compute_Shader", .MultiSample = 1},
            {.Key = "AJA_YCbCr2RGB_Pass", .Shader = "AJA_YCbCr2RGB_Shader", .MultiSample = 1},
            {.Key = "AJA_RGB2YCbCr_Pass", .Shader = "AJA_RGB2YCbCr_Shader", .MultiSample = 1}
        };

        *outCount = sizeof(passes) / sizeof(passes[0]);

        if (!outMzPassInfos)
            return MZ_RESULT_SUCCESS;

        for (auto s : passes)
            *outMzPassInfos++ = s;

        return MZ_RESULT_SUCCESS;
    }

    static MzResult GetShaderSource(MzBuffer * outSpirvBuf) 
    { 
        return MZ_RESULT_SUCCESS;
    }

    static bool CanCreateNode(const MzFbNode * node) 
    { 
        for (auto pin : *node->pins())
        {
            if (pin->name()->str() == "Device")
            {
                if (flatbuffers::IsFieldPresent(pin, mz::fb::Pin::VT_DATA))
                    return AJADevice::DeviceAvailable((char *)pin->data()->Data(),
                                                      node->class_name()->str() == "AJA.AJAIn");
                break;
            }
        }
        return AJADevice::GetAvailableDevice(node->class_name()->str() == "AJA.AJAIn");
    }
    
    static void OnNodeCreated(const MzFbNode * inNode, void** ctx) 
    { 
        auto& node = *inNode;
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

            mzEngine.HandleEvent(CreateAppEvent(
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
        mzEngine.HandleEvent(
            CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->Mapping.NodeId, ClearFlags::NONE, &pinsToDel,
                                                                  &pinsToAdd, 0, 0, 0, 0, &msg)));
    }


    static void OnNodeUpdated(void* ctx, const MzFbNode * node) 
    {
        ((AJAClient *)ctx)->OnNodeUpdate(*node);
    }
    
    static void OnNodeDeleted(void* ctx, const MzUUID nodeId) 
    { 
        auto c = ((AJAClient *)ctx);
        c->OnNodeRemoved();
        delete c;
    }
    static void OnPinValueChanged(void* ctx, const MzUUID id, MzBuffer * value)
    { 
        return ((AJAClient *)ctx)->OnPinValueChanged(id, value->Data);
    }

    static void OnPinConnected(void* ctx, const MzUUID pinId) { }
    static void OnPinDisconnected(void* ctx, const MzUUID pinId) { }

    static void OnPinShowAsChanged(void* ctx, const MzUUID id, MzFbShowAs showAs) 
    { 
    }

    static void OnNodeSelected(const MzUUID graphId, const MzUUID selectedNodeId) { }

    static void OnPathCommand(void* ctx, const MzPathCommand* params)
    { 
        auto aja = ((AJAClient *)ctx);
        aja->OnPathCommand(params->Id, (app::PathCommand)params->CommandType, mz::Buffer((u8*)params->Args.Data, params->Args.Size));
    }

    static MzResult GetFunctions(size_t * outCount, const char** pName, PFN_NodeFunctionExecute * outFunction) 
    {
        return MZ_RESULT_SUCCESS;
    }
    static bool  ExecuteNode(void* ctx, const MzNodeExecuteArgs * args) { return MZ_RESULT_SUCCESS; }
    static bool  CanCopy(void* ctx, MzCopyInfo * copyInfo) 
    { 
        return MZ_RESULT_SUCCESS;
    }

    static bool  BeginCopyFrom(void* ctx, MzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->BeginCopyTo(*cpy); 
    }

    static bool  BeginCopyTo(void* ctx, MzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->BeginCopyTo(*cpy); 
    }

    static void  EndCopyFrom(void* ctx, MzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->EndCopyFrom(*cpy);
    }

    static void  EndCopyTo(void* ctx, MzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->EndCopyTo(*cpy);
    }

    static void OnMenuRequested(void* ctx, const MzContextMenuRequest * request) 
    { 
        ((AJAClient *)ctx)->OnMenuFired(*request);
    }

    static void OnMenuCommand(void* ctx, uint32_t cmd) 
    { 
        ((AJAClient *)ctx)->OnCommandFired(cmd); 
    }

    static void OnKeyEvent(void* ctx, const MzKeyEvent * keyEvent) { }
};

void MZAPI_ATTR RegisterAJA(NodeActionsMap& functions)
{
    auto &actions = functions["AJA.AJAIn"];

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

            mzEngine.Log((aja->Device->GetDisplayName() + " SingleLink " + std::to_string(i + 1) + " info").c_str(),
						 ss.str().c_str());
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

    
        mzEngine.RegisterShader("AJA_YCbCr2RGB_Shader", Blob2Buf(YCbCr2RGB));
        mzEngine.RegisterShader("AJA_YCbCr2RGB_Shader", Blob2Buf(YCbCr2RGB));
        mzEngine.RegisterShader("AJA_RGB2YCbCr_Shader", Blob2Buf(RGB2YCbCr));
        mzEngine.RegisterShader("AJA_RGB2YCbCr_Compute_Shader", Blob2Buf(RGB2YCbCr2));
        mzEngine.RegisterShader("AJA_YCbCr2RGB_Compute_Shader", Blob2Buf(YCbCr2RGB2));

        mzEngine.RegisterPass2({.Key = "AJA_RGB2YCbCr_Compute_Pass",.Shader="AJA_RGB2YCbCr_Compute_Shader"});
        mzEngine.RegisterPass2({.Key = "AJA_YCbCr2RGB_Compute_Pass",.Shader="AJA_YCbCr2RGB_Compute_Shader"});
        mzEngine.RegisterPass2({.Key = "AJA_YCbCr2RGB_Pass",.Shader="AJA_YCbCr2RGB_Shader"});
        mzEngine.RegisterPass2({.Key = "AJA_RGB2YCbCr_Pass",.Shader="AJA_RGB2YCbCr_Shader"});
        
        for (auto c : AJAClient::Ctx.Clients)
            for (auto& p : c->Pins)
                p->StartThread();
    };

    functions["AJA.AJAOut"] = actions;
}

extern "C"
{

MZAPI_ATTR MzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, MzNodeFunctions* outFunctions)
{
    *outSize = 2;
    if (!outFunctions)
        return MZ_RESULT_SUCCESS;
   
    outFunctions[0] = outFunctions[1] = {
        .CanCreateNode = AJA::CanCreateNode,
        .OnNodeCreated = AJA::OnNodeCreated,
        .OnNodeUpdated = AJA::OnNodeUpdated,
        .OnNodeDeleted = AJA::OnNodeDeleted,
        .OnPinValueChanged = AJA::OnPinValueChanged,
        .OnPinConnected = AJA::OnPinConnected,
        .OnPinDisconnected = AJA::OnPinDisconnected,
        .OnPinShowAsChanged = AJA::OnPinShowAsChanged,
        .OnNodeSelected = AJA::OnNodeSelected,
        .OnPathCommand = AJA::OnPathCommand,
        .GetFunctions = AJA::GetFunctions,
        .ExecuteNode = AJA::ExecuteNode,
        .CanCopy = AJA::CanCopy,
        .BeginCopyFrom = AJA::BeginCopyFrom,
        .BeginCopyTo = AJA::BeginCopyTo,
        .EndCopyFrom = AJA::EndCopyFrom,
        .EndCopyTo = AJA::EndCopyTo,
        .GetShaderSource = AJA::GetShaderSource,
        .GetShaders = AJA::GetShaders,
        .GetPasses = AJA::GetPasses,
        .OnMenuRequested = AJA::OnMenuRequested,
        .OnMenuCommand = AJA::OnMenuCommand,
        .OnKeyEvent = AJA::OnKeyEvent,
    };

    outFunctions[0].TypeName = "AJA.AJAIn";
    outFunctions[1].TypeName = "AJA.AJAOut";

    return MZ_RESULT_SUCCESS;
}

}

} // namespace mz
