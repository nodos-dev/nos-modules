// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/PluginAPI.h>
#include <MediaZ/Helpers.hpp>

MZ_INIT()

#include "Common_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "module.Curve/ZD.Curve.h"

#include <fstream>
#include <filesystem>

using namespace ZD::Curve;

namespace mz
{

MZ_REGISTER_NAME(JSON);
MZ_REGISTER_NAME(Zoom);
MZ_REGISTER_NAME(Focus);
MZ_REGISTER_NAME(FOV);
MZ_REGISTER_NAME(k1k2);
MZ_REGISTER_NAME_SPACED(Focal_Length_Offset, "Focal Length Offset");
MZ_REGISTER_NAME_SPACED(Nodal_Point, "Nodal Point");
MZ_REGISTER_NAME_SPACED(Focus_Distance_Curve, "Focus Distance Curve");
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
        mzEngine.HandleEvent(CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &msg)));
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
            mzEngine.LogE("Failed to parse lens file", what);
            UpdateStatus(lastVal.empty() ? "No Lens File" : "Invalid Lens File");
        }
    }
    
    void Load(fb::Node const& node)
    {
       
        auto pins = PinMapping::Load(node);
        if(auto pin = pins[MZN_JSON])
        {
            if(flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
            {
                Deser((const char*)pin->data()->Data());
            }
        }
    }

    void ValueChanged(mz::Name pinName, void* val)
    {
        if(pinName == MZN_JSON)
        {
            Deser((const char*)val);
            return;
        }
		if (pinName == MZN_Focal_Length_Offset)
        {
            calibrator.FocalLengthOffset = *(f64*)val;
            return;
        }
    }
};

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions* outFunctions)
{
	if (!outFunctions)
	{
		*outSize = 1;
		return MZ_RESULT_SUCCESS;
	}
	auto& funcs = outFunctions[0];
	funcs.TypeName = MZ_NAME_STATIC("zd.frustum.Frustum");
	funcs.OnNodeCreated = [](auto *node,void **ctx) {
		Frustum* f = new Frustum();
		f->Load(*node);
		*ctx = f;
	};
	funcs.OnNodeUpdated = [](void *ctx, auto *node) {
		((Frustum*)ctx)->Load(*node);
	};
	funcs.OnNodeDeleted = [](void *ctx, mzUUID nodeId) {
		delete (Frustum*)ctx;
	};
	funcs.OnPinValueChanged = [](void *ctx, mzName pinName, mzBuffer *value) {
		((Frustum*)ctx)->ValueChanged(pinName, value->Data);
	};
	funcs.ExecuteNode = [](void* ctx, mzNodeExecuteArgs const* args) {
		auto values = GetPinValues(args);
		auto zoom = (f64*)values[MZN_Zoom];
		auto focus = (f64*)values[MZN_Focus];

		if (zoom && focus)
		{
			auto& clb = ((Frustum*)ctx)->calibrator;
			if (auto fov = (f64*)values[MZN_FOV])
				*fov = clb.GetFoV(*zoom, *focus);
			if (auto k1k2 = (mz::fb::vec2*)values[MZN_k1k2])
				(glm::vec2&)* k1k2 = clb.GetK1K2(*zoom, *focus);
			if (auto np = (f64*)values[MZN_Nodal_Point])
				*np = clb.NodalPointCurve.GetInterpolatedValue(*zoom);
			if (auto fdc = (f64*)values[MZN_Focus_Distance_Curve])
				*fdc = clb.FocusDistanceCurve.GetInterpolatedValue(*focus);
		}
		return MZ_RESULT_SUCCESS;
	};
	return MZ_RESULT_SUCCESS;
}

}

} // namespace mz