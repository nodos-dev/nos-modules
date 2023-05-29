// Copyright MediaZ AS. All Rights Reserved.

#include "Track.h"
#include "mzDefines.h"

#include <bit>

struct FZDStypeTimecode
{
    u8 Value1;
    u8 Value2;
    u8 Value3;

    i32 GetValue() const
    {
        return (Value1 << 16) | (Value2 << 8) | (Value3);
    }
};

struct StypeMsg_A5
{
    u8 Header = 0;
    u8 buf[52] = {};
    u8 Checksum = 0;

    glm::vec3 XYZ() const { return Get<glm::vec3, 0>() * 100.f; }
    glm::vec3 PTR() const { return Get<glm::vec3, 12>().zxy; }
    glm::vec2 Fov() const { return Get<glm::vec2, 24>(); }
    f32 Focus() const { return Get<f32, 28>(); }
    glm::vec2 K1K2() const { return Get<glm::vec2, 32>(); }
    glm::vec2 CenterShift() const { return Get<glm::vec2, 40>(); }

    template<class T, u8 offset>
    T Get() const
    {
        T val;
        memcpy(&val, &buf[offset], sizeof(T));
        return val;
    }


    StypeMsg_A5() = delete;

    bool IsChecksumOK() const
    {
        const u8 *Buffer = &Header;
        u8 TotalSum = 0;
        for (u32 Index = 0; Index < sizeof(StypeMsg_A5) - 1; ++Index)
        {
            TotalSum += Buffer[Index];
        }
        if (TotalSum != Checksum)
        {
            // auto message = FString::Printf(TEXT("Stype_A5 Checksum error: %d != %d"), Checksum, TotalSum);
            // ZD_LOG_WARNING(message);
            // IZDStudioInterface::Get().GetNodeGraph()->Log(FZDLogEntry(message));
            return false;
        }
        return true;
    }
};

struct StypeMsg_0F
{
    u8 Header;
    u8 Command = 0;
    FZDStypeTimecode Timecode;
    u8 MessageId = 0;
    u8 buf[60] = {};
    u8 Checksum;

    StypeMsg_0F() = delete;

    template<class T, u8 offset>
    T Get() const
    {
        T val;
        memcpy(&val, &buf[offset], sizeof(T));
        return val;
    }
    
    glm::vec3 XYZ() const { return Get<glm::vec3, 0>() * 100.f;  }
    glm::vec3 PTR() const { return Get<glm::vec3, 12>().zxy;  }
    f32 FovX() const { return Get<f32, 24>(); }
    f32 AspectRatio() const { return Get<f32, 28>(); }
    f32 Focus() const { return Get<f32, 32>(); }
    f32 Zoom() const { return Get<f32, 36>(); }
    glm::vec2 K1K2() const { return Get<glm::vec2, 40>(); }
    glm::vec2 CenterShift() const { return Get<glm::vec2, 48>(); }
    f32 PAWidth() const { return Get<f32, 56>(); }

    bool IsChecksumOK() const
    {
        const u8 *Buffer = &Header;
        u8 TotalSum = 0;
        for (u32 Index = 0; Index < sizeof(StypeMsg_0F) - 1; ++Index)
        {
            TotalSum += Buffer[Index];
        }
        if (TotalSum != Checksum)
        {
            // auto message = FString::Printf(TEXT("Stype_HF Checksum error: %d != %d"), Checksum, TotalSum);
            // ZD_LOG_WARNING(message);
            // IZDStudioInterface::Get().GetNodeGraph()->Log(FZDLogEntry(message));
            return false;
        }
        return true;
    }

};


auto what = offsetof(StypeMsg_0F, buf);
auto what2 = sizeof(StypeMsg_0F);

static_assert(
    offsetof(StypeMsg_0F, buf) == 
    (sizeof(u8) * 6)
);

namespace mz
{


struct Stype : public TrackNodeContext
{
    using TrackNodeContext::TrackNodeContext;

    bool Parse(std::vector<u8> const& buf, fb::TTrack& TrackData) override
    {
        TrackData.sensor_size.mutate_x(9.590f);
        TrackData.sensor_size.mutate_y(5.394f);
        TrackData.fov = 60.0;
        TrackData.distortion_scale = 1;
        TrackData.pixel_aspect_ratio = 1;

        switch (buf[0])
        {
        case 0x0F: {
            assert(sizeof(StypeMsg_0F) >= buf.size());
            auto ctx = (StypeMsg_0F *)buf.data();
            (glm::dvec3&)TrackData.location = ctx->XYZ();
            (glm::dvec3&)TrackData.rotation = ctx->PTR();
            (glm::dvec2&)TrackData.center_shift = ctx->CenterShift();
            TrackData.fov = ctx->FovX();
            TrackData.focus = ctx->Focus();
            TrackData.zoom = ctx->Zoom();
            TrackData.sensor_size = fb::vec2d(ctx->PAWidth(), ctx->PAWidth() / ctx->AspectRatio());
            (glm::dvec2 &)TrackData.k1k2 = ctx->K1K2();

            break;
        }
        case 0xA5: {
            auto ctx = (StypeMsg_A5 *)buf.data();
            (glm::dvec3 &)TrackData.location = ctx->XYZ();
            (glm::dvec3 &)TrackData.rotation = ctx->PTR();
            (glm::dvec2 &)TrackData.center_shift = ctx->CenterShift();
            TrackData.fov = ctx->Fov().x;
            TrackData.focus = ctx->Focus();
            (glm::dvec2 &)TrackData.k1k2 = ctx->K1K2();
            break;
        }
        }
        
		const f64 SensorAspectRatio = TrackData.sensor_size.x() / TrackData.sensor_size.y();
		const f64 SensorHalfWidth   = TrackData.sensor_size.x()/ 2.;
		const f64 RD2 = SensorHalfWidth * SensorHalfWidth + (SensorHalfWidth / SensorAspectRatio) * (SensorHalfWidth / SensorAspectRatio);
		const f64 RD4 = RD2 * RD2;

        TrackData.k1k2.mutate_x(TrackData.k1k2.x() * RD2);
        TrackData.k1k2.mutate_y(TrackData.k1k2.y() * RD4);

        return true;
    }

    ~Stype()
    {
        if (IsRunning())
        {
            Stop();
        }
    }
};

void RegisterStypeNode(MzNodeFunctions &functions)
{
	functions.TypeName = "mz.track.Stype";
    RegisterTrackCommon(functions);

	functions.OnNodeCreated = [](fb::Node const* node, void** ctx) {
		auto context = new Stype();
		*ctx = context;
		auto pins = context->Load(*node);
		if (auto pin = pins["UDP_Port"])
		{
			if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
			{
				context->Port = *(uint16_t*)pin->data()->data();
			}
		}
		if (auto pin = pins["Enable"])
		{
			if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
			{
				if (*(bool*)pin->data()->data())
				{
					context->Start();
				}
			}
		}
    };

	functions.OnPinValueChanged = [](auto ctx, auto id, MzBuffer* value) 
	{ 
		Stype* fnctx = (Stype*)ctx;
		fnctx->OnPinValueChanged(id, value->Data);
	};
	functions.OnNodeDeleted = [](auto ctx, auto id) {
		delete (Stype*)ctx;
	};
}

} // namespace mz