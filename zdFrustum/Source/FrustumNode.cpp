// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Plugin.h>

#include "Common_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "module.Curve/ZD.Curve.h"

#include <fstream>
#include <filesystem>

using namespace ZD::Curve;

namespace mz
{

EngineNodeServices GServices;

struct Frustum: PinMapping
{
    GenericCalibration calibrator;
    std::string lastVal;
    
    void UpdateStatus(const char* str)
    {
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> msg = { 
            fb::CreateNodeStatusMessageDirect(fbb, str, fb::NodeStatusMessageType::WARNING) 
        };
        GServices.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
    }

    void Deser(const char* val)
    {
        if(lastVal == val) return;
        lastVal = val;
        try
        {
            calibrator = {};
            calibrator.Deserialize(ReadToString(val));
            UpdateStatus(std::filesystem::path(val).stem().string().c_str());
        }
        catch (std::exception& e)
        {
            auto what = e.what();
            GServices.LogE("Failed to parse lens file", what);
            UpdateStatus(lastVal.empty() ? "No Lens File" : "Invalid Lens File");
        }
    }
    
    void Load(fb::Node const& node)
    {
        auto pins = PinMapping::Load(node);
        if(auto pin = pins["JSON"])
        {
            if(flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
            {
                Deser((const char*)pin->data()->Data());
            }
        }
    }

    void ValueChanged(mz::fb::UUID id, void* val)
    {
        auto name = GetPinName(id);
        if(name == "JSON")
        {
            Deser((const char*)val);
            return;
        }
        if(name == "Focal Length Offset")
        {
            calibrator.FocalLengthOffset = *(f64*)val;
            return;
        }
    }
};

extern "C"
{
void MZAPI_ATTR Register(NodeActionsMap& functions, EngineNodeServices services, std::set<flatbuffers::Type const*> const& types)
{
    GServices = services;
    auto &actions = functions["ZD.Frustum"];
    actions.NodeCreated = [](auto &node, Args &args, void **ctx) 
    { 
        Frustum* f = new Frustum();
        f->Load(node);

        *ctx = f;
     };

    actions.NodeUpdate = [](auto &node, auto ctx) 
    {
        ((Frustum*)ctx)->Load(node);
    };
    
    actions.MenuFired = [](auto &node, auto ctx, auto &request) {};
    actions.CommandFired = [](auto &node, auto ctx, u32 cmd) {};

    actions.NodeRemoved = [](auto ctx, auto &id) {
        delete (Frustum*)ctx;
    };

    actions.PinValueChanged = [](auto ctx, auto &id, mz::Buffer* value) {
        ((Frustum*)ctx)->ValueChanged(id, value->data());
    };

    actions.PinShowAsChanged = [](auto ctx, auto &id, int value) {};
    actions.EntryPoint = [](mz::Args &pins, auto ctx) { 
        
        auto zoom  = pins.Get<f64>("Zoom");
        auto focus = pins.Get<f64>("Focus");
        if(zoom && focus)
        {
            auto& clb = ((Frustum*)ctx)->calibrator;
            if(auto fov = pins.Get<f64>("FOV"))                    *fov = clb.GetFoV(*zoom, *focus);
            if (auto k1k2 = pins.Get<mz::fb::vec2>("k1k2")) (glm::vec2&)*k1k2 = clb.GetK1K2(*zoom, *focus);
            if(auto np = pins.Get<f64>("Nodal Point"))             *np =  clb.NodalPointCurve.GetInterpolatedValue(*zoom);
            if(auto fdc = pins.Get<f64>("Focus Distance Curve"))   *fdc =  clb.FocusDistanceCurve.GetInterpolatedValue(*focus);
        }
        
        return true; 
    };
}
}

} // namespace mz