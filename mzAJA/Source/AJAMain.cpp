// Copyright MediaZ AS. All Rights Reserved.

#include "AJAMain.h"
#include "CopyThread.h"
#include "AJAClient.h"

#include "RGB2YCbCr.comp.spv.dat"
#include "RGB2YCbCr.frag.spv.dat"
#include "YCbCr2RGB.comp.spv.dat"
#include "YCbCr2RGB.frag.spv.dat"

#include <MediaZ/PluginAPI.h>

using namespace mz;

MZ_INIT();

namespace mz
{

static mzBuffer Blob2Buf(std::vector<u8> const& v) 
{ 
    return { (void*)v.data(), v.size() }; 
};

static std::vector<std::pair<Name, std::vector<u8>>> GShaders;

struct AJA
{
    static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders)
    {
        *outCount = GShaders.size();
        if(!outShaders) 
            return MZ_RESULT_SUCCESS;

        for (auto& [name, spirv] : GShaders)
        {
            *outShaders++ = {
                .Key = name,
                .SpirvBlob = { spirv.data(), spirv.size() },
            };
        }

        return MZ_RESULT_SUCCESS;
    };

    static mzResult GetPasses(size_t* outCount, mzPassInfo* outMzPassInfos)
    {
        const std::vector<mzPassInfo> passes =
        {
			{.Key = MZN_AJA_RGB2YCbCr_Compute_Pass, .Shader = MZN_AJA_RGB2YCbCr_Compute_Shader, .MultiSample = 1},
			{.Key = MZN_AJA_YCbCr2RGB_Compute_Pass, .Shader = MZN_AJA_YCbCr2RGB_Compute_Shader, .MultiSample = 1},
			{.Key = MZN_AJA_YCbCr2RGB_Pass, .Shader = MZN_AJA_YCbCr2RGB_Shader, .MultiSample = 1},
			{.Key = MZN_AJA_RGB2YCbCr_Pass, .Shader = MZN_AJA_RGB2YCbCr_Shader, .MultiSample = 1}
        };

        *outCount = passes.size();

        if (!outMzPassInfos)
            return MZ_RESULT_SUCCESS;
        for (auto& pass : passes)
        {
            *outMzPassInfos++ = pass;
        }

        return MZ_RESULT_SUCCESS;
    }

    static mzResult GetShaderSource(mzBuffer * outSpirvBuf) 
    { 
        return MZ_RESULT_SUCCESS;
    }

    static mzResult CanCreateNode(const mzFbNode * node) 
    { 
        for (auto pin : *node->pins())
        {
            if (pin->name()->str() == "Device")
            {
                if (flatbuffers::IsFieldPresent(pin, mz::fb::Pin::VT_DATA))
                    return AJADevice::DeviceAvailable((char *)pin->data()->Data(),
                                                      node->class_name()->str() == "AJA.AJAIn") ? MZ_RESULT_SUCCESS : MZ_RESULT_FAILED;
                break;
            }
        }
		if (AJADevice::GetAvailableDevice(node->class_name()->str() == "AJA.AJAIn"))
        {
            return MZ_RESULT_SUCCESS;
        }
        return MZ_RESULT_FAILED;
    }
    
    static void OnNodeCreated(const mzFbNode * inNode, void** ctx) 
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
            const auto str = dev->GetDisplayName() + "-AJAOut-Reference-Source";
            flatbuffers::FlatBufferBuilder fbb;
            std::vector<std::string> list{"Reference In", "Free Run"};
            for (int i = 1; i <= NTV2DeviceGetNumVideoInputs(dev->ID); ++i)
            {
                list.push_back("SDI In " + std::to_string(i));
            }
            
            mzEngine.HandleEvent(CreateAppEvent(
                fbb, mz::app::CreateUpdateStringList(fbb, mz::fb::CreateStringList(fbb, fbb.CreateString(str), fbb.CreateVectorOfStrings(list)))));
        }

        AJAClient *c = new AJAClient(isIn, dev);
        *ctx = c;

        std::string refSrc = "Ref : " + NTV2ReferenceSourceToString(c->Ref, true) + " - " + NTV2FrameRateToString(c->FR, true);
        flatbuffers::FlatBufferBuilder fbb;

        std::vector<flatbuffers::Offset<mz::fb::Pin>> pinsToAdd;
        std::vector<::flatbuffers::Offset<mz::PartialPinUpdate>> pinsToUpdate;
        using mz::fb::ShowAs;
        using mz::fb::CanShowAs;

        PinMapping mapping;
        auto loadedPins = mapping.Load(node);

        AddIfNotFound(MZN_Device,
					  "string",
					  StringValue(dev->GetDisplayName()),
					  loadedPins,
					  pinsToAdd, pinsToUpdate,
					  fbb,
                      ShowAs::PROPERTY, CanShowAs::PROPERTY_ONLY);
		if (auto val = AddIfNotFound(MZN_Dispatch_Size,
									 "mz.fb.vec2u",
									 mz::Buffer::From(mz::fb::vec2u(c->DispatchSizeX, c->DispatchSizeY)),
                                     loadedPins, pinsToAdd, pinsToUpdate, fbb))
        {
            c->DispatchSizeX = ((glm::uvec2 *)val)->x;
            c->DispatchSizeY = ((glm::uvec2 *)val)->y;
        }

        if (auto val = AddIfNotFound(
				MZN_Shader_Type, "AJA.Shader", mz::Buffer::From(ShaderType(c->Shader)), loadedPins,
                                     pinsToAdd, pinsToUpdate, fbb))
        {
            c->Shader = *((ShaderType *)val);
        }

        if (auto val =
				AddIfNotFound(MZN_Debug, "uint", mz::Buffer::From(u32(c->Debug)), loadedPins, pinsToAdd, pinsToUpdate, fbb))
        {
            c->Debug = *((u32 *)val);
        }

        if (!isIn)
        {
            mz::fb::TVisualizer vis = { .type = mz::fb::VisualizerType::COMBO_BOX, .name = dev->GetDisplayName() + "-AJAOut-Reference-Source" };
            if (auto ref = AddIfNotFound(MZN_ReferenceSource,
                "string",
                StringValue(refSrc),
                loadedPins,
                pinsToAdd, pinsToUpdate,
                fbb,
                ShowAs::PROPERTY, CanShowAs::PROPERTY_ONLY, vis))
            {
                refSrc = (char *)ref;
            }
        }
        c->SetReference(refSrc);

        std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg;
        std::vector<mz::fb::UUID> pinsToDel;
        c->OnNodeUpdate(std::move(mapping), loadedPins, pinsToDel);
        c->UpdateStatus(fbb, msg);
        mzEngine.HandleEvent(
            CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &c->Mapping.NodeId, ClearFlags::NONE, &pinsToDel,
                                                                  &pinsToAdd, 0, 0, 0, 0, &msg, &pinsToUpdate)));
    }


    static void OnNodeUpdated(void* ctx, const mzFbNode * node) 
    {
        ((AJAClient *)ctx)->OnNodeUpdate(*node);
    }
    
    static void OnNodeDeleted(void* ctx, const mzUUID nodeId) 
    { 
        auto c = ((AJAClient *)ctx);
        c->OnNodeRemoved();
        delete c;
    }
    static void OnPinValueChanged(void* ctx, const mzName pinName, mzBuffer * value)
    { 
        return ((AJAClient *)ctx)->OnPinValueChanged(pinName, value->Data);
    }

    static void OnPathCommand(void* ctx, const mzPathCommand* params)
    { 
        auto aja = ((AJAClient *)ctx);
        aja->OnPathCommand(params->PinId, (app::PathCommand)params->Command, mz::Buffer((u8*)params->Args.Data, params->Args.Size));
    }

    static void ReloadShaders(void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs)
    {
		std::string workFolder = mzEngine.WorkFolder();
		std::string sources = workFolder + "/../../plugins/mzAJA/Source";
		std::string tempOutPrefix = workFolder + "/../../plugins/mzAJA/Source";
        auto genSpirv = [](std::string const& workFolder,
							std::string const& sources,
							std::string const& intermediateFolder,
							std::string const& shaderName) {
			system(
				("glslc -O -g " + sources + "/" + shaderName + " -c -o " + intermediateFolder + "/" + shaderName + ".spv.temp")
					.c_str());
			system(("spirv-opt -O " + intermediateFolder + "/" + shaderName + ".spv.temp" + " -o " + intermediateFolder +
					"/" + shaderName + ".spv.opt.temp")
					   .c_str());
			std::remove((intermediateFolder + "/" + shaderName + ".spv.temp").c_str());
			auto spirv = ReadSpirv(intermediateFolder + "/" + shaderName + ".spv.opt.temp");
			std::remove((intermediateFolder + "/" + shaderName + ".spv.opt.temp").c_str());
			return spirv;
        };

        auto YCbCr2RGBFrag = genSpirv(workFolder, sources, tempOutPrefix, "YCbCr2RGB.frag");
		auto RGB2YCbCrFrag = genSpirv(workFolder, sources, tempOutPrefix, "RGB2YCbCr.frag");
		auto RGB2YCbCrComp = genSpirv(workFolder, sources, tempOutPrefix, "RGB2YCbCr.comp");
		auto YCbCr2RGBComp = genSpirv(workFolder, sources, tempOutPrefix, "YCbCr2RGB.comp");

        for (auto c : AJAClient::Ctx.Clients)
            for (auto& p : c->Pins)
                p->Stop();

        GShaders[0] = {MZN_AJA_RGB2YCbCr_Compute_Shader, RGB2YCbCrComp};
		GShaders[1] = {MZN_AJA_YCbCr2RGB_Compute_Shader, YCbCr2RGBComp};
		GShaders[2] = {MZN_AJA_RGB2YCbCr_Shader, RGB2YCbCrFrag};
		GShaders[3] = {MZN_AJA_YCbCr2RGB_Shader, YCbCr2RGBFrag};
        
        mzEngine.ReloadShaders(((AJAClient*)ctx)->Input ? MZN_AJA_AJAIn : MZN_AJA_AJAOut);
                 
        for (auto c : AJAClient::Ctx.Clients)
            for (auto& p : c->Pins)
            {
                p->CreateRings(p->GetRingSize());
                p->StartThread();
            }
    }

    static void PathRestart(void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs)
    {
        auto aja = ((AJAClient *)ctx);
        for (auto& th : aja->Pins)
            th->PathRestart();
    }

    static mzResult GetFunctions(size_t * outCount, mzName* outName, mzPfnNodeFunctionExecute* outFunction) 
    {
        *outCount = 2;
        if(!outName || !outFunction)
            return MZ_RESULT_SUCCESS;
        outFunction[0] = ReloadShaders;
		outName[0] = MZ_NAME_STATIC("ReloadShaders");
        outFunction[1] = PathRestart;
        outName[1] = MZ_NAME_STATIC("PathRestart");
        return MZ_RESULT_SUCCESS;
	    // auto &actions = functions["AJA.AJAIn"];
	    //
	    // actions.NodeFunctions["DumpInfo"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
	    //     auto aja = ((AJAClient *)ctx);
	    //
	    //     for (u32 i = 0; i < 4; ++i)
	    //     {
	    //         AJALabelValuePairs info = {};
	    //         aja->Device->GetVPID(NTV2Channel(i)).GetInfo(info);
	    //         std::ostringstream ss;
	    //         for (auto &[k, v] : info)
	    //         {
	    //             ss << k << " : " << v << "\n";
	    //         }
	    //
	    //         mzEngine.Log((aja->Device->GetDisplayName() + " SingleLink " + std::to_string(i + 1) + " info").c_str(),
					// 		 ss.str().c_str());
	    //     }
	    // };
	    //
	    // actions.NodeFunctions["StartLog"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
	    //
	    // };
	    //
	    // actions.NodeFunctions["StopLog"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
	    //
	    // };
	    //
	    // actions.NodeFunctions["ReloadShaders"] = [](mz::Args &pins, mz::Args &functionParams, void *ctx) {
	    //     system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/YCbCr2RGB.frag -c -o " MZ_REPO_ROOT
	    //            "/../YCbCr2RGB_.frag");
	    //     system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/RGB2YCbCr.frag -c -o " MZ_REPO_ROOT
	    //            "/../RGB2YCbCr_.frag");
	    //     system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/RGB2YCbCr.comp -c -o " MZ_REPO_ROOT
	    //            "/../RGB2YCbCr_.comp");
	    //     system("glslc -O -g " MZ_REPO_ROOT "/Plugins/mzBasic/Source/AJA/YCbCr2RGB.comp -c -o " MZ_REPO_ROOT
	    //            "/../YCbCr2RGB_.comp");
	    //
	    //     system("spirv-opt -O " MZ_REPO_ROOT "/../YCbCr2RGB_.frag -o " MZ_REPO_ROOT "/../YCbCr2RGB.frag");
	    //     system("spirv-opt -O " MZ_REPO_ROOT "/../RGB2YCbCr_.frag -o " MZ_REPO_ROOT "/../RGB2YCbCr.frag");
	    //     system("spirv-opt -O " MZ_REPO_ROOT "/../RGB2YCbCr_.comp -o " MZ_REPO_ROOT "/../RGB2YCbCr.comp");
	    //     system("spirv-opt -O " MZ_REPO_ROOT "/../YCbCr2RGB_.comp -o " MZ_REPO_ROOT "/../YCbCr2RGB.comp");
	    //     auto YCbCr2RGB = ReadSpirv(MZ_REPO_ROOT "/../YCbCr2RGB.frag");
	    //     auto RGB2YCbCr = ReadSpirv(MZ_REPO_ROOT "/../RGB2YCbCr.frag");
	    //     auto RGB2YCbCr2 = ReadSpirv(MZ_REPO_ROOT "/../RGB2YCbCr.comp");
	    //     auto YCbCr2RGB2 = ReadSpirv(MZ_REPO_ROOT "/../YCbCr2RGB.comp");
	    //
	    //     for (auto c : AJAClient::Ctx.Clients)
	    //         for (auto& p : c->Pins)
	    //             p->Stop();
	    //
	    //
	    //     mzEngine.RegisterShader("AJA_YCbCr2RGB_Shader", Blob2Buf(YCbCr2RGB));
	    //     mzEngine.RegisterShader("AJA_YCbCr2RGB_Shader", Blob2Buf(YCbCr2RGB));
	    //     mzEngine.RegisterShader("AJA_RGB2YCbCr_Shader", Blob2Buf(RGB2YCbCr));
	    //     mzEngine.RegisterShader("AJA_RGB2YCbCr_Compute_Shader", Blob2Buf(RGB2YCbCr2));
	    //     mzEngine.RegisterShader("AJA_YCbCr2RGB_Compute_Shader", Blob2Buf(YCbCr2RGB2));
	    //
	    //     mzEngine.RegisterPass2({.Key = "AJA_RGB2YCbCr_Compute_Pass",.Shader="AJA_RGB2YCbCr_Compute_Shader"});
	    //     mzEngine.RegisterPass2({.Key = "AJA_YCbCr2RGB_Compute_Pass",.Shader="AJA_YCbCr2RGB_Compute_Shader"});
	    //     mzEngine.RegisterPass2({.Key = "AJA_YCbCr2RGB_Pass",.Shader="AJA_YCbCr2RGB_Shader"});
	    //     mzEngine.RegisterPass2({.Key = "AJA_RGB2YCbCr_Pass",.Shader="AJA_RGB2YCbCr_Shader"});
	    //     
	    //     for (auto c : AJAClient::Ctx.Clients)
	    //         for (auto& p : c->Pins)
	    //             p->StartThread();
	    // };
	    //
	    // functions["AJA.AJAOut"] = actions;
        return MZ_RESULT_SUCCESS;
    }
    static mzResult  ExecuteNode(void* ctx, const mzNodeExecuteArgs * args) { return MZ_RESULT_SUCCESS; }
    static mzResult  CanCopy(void* ctx, mzCopyInfo * copyInfo)
    { 
        return MZ_RESULT_SUCCESS;
    }

    static mzResult  BeginCopyFrom(void* ctx, mzCopyInfo * cpy)
    { 
        if (((AJAClient*)ctx)->BeginCopyFrom(*cpy))
        {
            return MZ_RESULT_SUCCESS;
        }
        return MZ_RESULT_FAILED;
    }

    static mzResult  BeginCopyTo(void* ctx, mzCopyInfo * cpy)
    { 
        if (((AJAClient*)ctx)->BeginCopyTo(*cpy))
        {
            return MZ_RESULT_SUCCESS;
        }
        return MZ_RESULT_FAILED;
    }

    static void  EndCopyFrom(void* ctx, mzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->EndCopyFrom(*cpy);
    }

    static void  EndCopyTo(void* ctx, mzCopyInfo * cpy)
    { 
        return ((AJAClient *)ctx)->EndCopyTo(*cpy);
    }

    static void OnMenuRequested(void* ctx, const mzContextMenuRequest * request) 
    { 
        ((AJAClient *)ctx)->OnMenuFired(*request);
    }

    static void OnMenuCommand(void* ctx, uint32_t cmd) 
    { 
        ((AJAClient *)ctx)->OnCommandFired(cmd); 
    }

    static void OnKeyEvent(void* ctx, const mzKeyEvent * keyEvent) { }
};

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions** outList)
{
    *outSize = 2;
    if (!outList)
        return MZ_RESULT_SUCCESS;
    auto* ajaIn = outList[0];
    auto* ajaOut = outList[1];
    ajaIn->TypeName = MZN_AJA_AJAIn;
    ajaOut->TypeName = MZN_AJA_AJAOut;
    ajaIn->CanCreateNode = ajaOut->CanCreateNode = AJA::CanCreateNode;
    ajaIn->OnNodeCreated = ajaOut->OnNodeCreated = AJA::OnNodeCreated;
    ajaIn->OnNodeUpdated = ajaOut->OnNodeUpdated = AJA::OnNodeUpdated;
    ajaIn->OnNodeDeleted = ajaOut->OnNodeDeleted = AJA::OnNodeDeleted;
    ajaIn->OnPinValueChanged = ajaOut->OnPinValueChanged = AJA::OnPinValueChanged;
    ajaIn->OnPathCommand = ajaOut->OnPathCommand = AJA::OnPathCommand;
    ajaIn->GetFunctions = ajaOut->GetFunctions = AJA::GetFunctions;
    ajaIn->ExecuteNode = ajaOut->ExecuteNode = AJA::ExecuteNode;
    ajaIn->CanCopy = ajaOut->CanCopy = AJA::CanCopy;
    ajaIn->BeginCopyFrom = ajaOut->BeginCopyFrom = AJA::BeginCopyFrom;
    ajaIn->BeginCopyTo = ajaOut->BeginCopyTo = AJA::BeginCopyTo;
    ajaIn->EndCopyFrom = ajaOut->EndCopyFrom = AJA::EndCopyFrom;
    ajaIn->EndCopyTo = ajaOut->EndCopyTo = AJA::EndCopyTo;
    ajaIn->GetShaderSource = ajaOut->GetShaderSource = AJA::GetShaderSource;
    ajaIn->GetShaders = ajaOut->GetShaders = AJA::GetShaders;
    ajaIn->GetPasses = ajaOut->GetPasses = AJA::GetPasses;
    ajaIn->OnMenuRequested = ajaOut->OnMenuRequested = AJA::OnMenuRequested;
    ajaIn->OnMenuCommand = ajaOut->OnMenuCommand = AJA::OnMenuCommand;
    ajaIn->OnKeyEvent = ajaOut->OnKeyEvent = AJA::OnKeyEvent;

    GShaders = {
		{MZN_AJA_RGB2YCbCr_Compute_Shader, {std::begin(RGB2YCbCr_comp_spv), std::end(RGB2YCbCr_comp_spv)}},
		{MZN_AJA_YCbCr2RGB_Compute_Shader, {std::begin(YCbCr2RGB_comp_spv), std::end(YCbCr2RGB_comp_spv)}},
		{MZN_AJA_RGB2YCbCr_Shader, {std::begin(RGB2YCbCr_frag_spv), std::end(RGB2YCbCr_frag_spv)}},
		{MZN_AJA_YCbCr2RGB_Shader, {std::begin(YCbCr2RGB_frag_spv), std::end(YCbCr2RGB_frag_spv)}},
	};

    return MZ_RESULT_SUCCESS;
}

}

} // namespace mz
