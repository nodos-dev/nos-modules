// Copyright MediaZ AS. All Rights Reserved.

#include "Track.h"
#include <bit>

constexpr u8 MAGIC_NUMBER = 0xF4;

namespace mz
{

struct MoSysAxis
{
    /*
        Angle:           24 bit signed int in degrees/1000
        Linear Position: 24 bit signed int in mm/100
        Lens Position:   16 bit unsigned int
        Lens Parameter:  32 bit float
    */

    enum XID: u8
    {
        PAN =                    0x01,
        TILT =                   0x02,
        ROLL =                   0x13,

        FOCUS =                  0x03,
        ZOOM =                   0x04,

        X =                      0x05,
        Y =                      0x06,
        Z =                      0x07,

        CX =                     0x54,
        CY =                     0x55,

        ENTRANCE_PUPIL =         0x50,
        LENS_DISTORTION_K1 =     0x51,
        LENS_DISTORTION_K2 =     0x52,
        FOCAL_LENGTH_FX =        0x53,
        FOCAL_DISTANCE =         0X57,
        TIMECODE =               0xF8,
        GENLOCK_INFO =           0xFF,

        CPAN =                   0x08,
        CTILT =                  0x09,
        TURNT =                  0x0A,
        CRANEX =                 0x0B,
        ORIENT =                 0x0C,
        IRIS =                   0x0D,
        DIGIFOCUS =              0x0E,
        DIGIZOOM =               0x0F,
        DIGIIRIS =               0x10,
        STRINGX =                0x11,
        STRINGY =                0x12,
        
        HEAD_Y =                 0x14,
        HEAD_Z =                 0x15,
        RAWPAN =                 0x32,
        RAWTILT =                0x33,
        RAWROLL =                0x34,
        RAWJIB =                 0x35,
        RAWTURN =                0x36,
        RAWTRACK =               0x37,
        RAWTELE =                0x38,
        RAWELEV =                0x39,
        VIRTUAL_FX =             0X56,
        GATEWAY_BUILD_VERSION =  0xFA,
        GATEWAY_BUILD_DATE =     0xFB,
        AXIS_DATA_INFO =         0xFC,
        NETWORK_STATUS_INFO =    0xFD,
        PACKET_TIMING_INFO =     0xFE,
        TRACKING_STATUS =     0xF9,

    } ID;
    u8 AxisStatus;
    u8 Position[3];

    enum Type : u8
    {
        UNKNOWN,
        S24,
        U24,
        S16,
        U16,
        F32,
        STATUS,
    };


    Type GetType() const
    {
        switch (ID)
        {
        case TRACKING_STATUS:
            return STATUS;
        case PAN:
        case TILT:
        case ROLL:
        case X:
        case Y:
        case Z:
            return S24;
        case FOCUS:
        case ZOOM:
            return U16;
        case CX:
        case CY:
        case LENS_DISTORTION_K1:
        case LENS_DISTORTION_K2:
        case FOCAL_DISTANCE:
        case FOCAL_LENGTH_FX:
        case VIRTUAL_FX:
            return F32;
        default:  return UNKNOWN;
        }
    }

    u16 Get16() const
    {
        return (Position[1] << 8) | Position[2];
    }

    u32 GetU() const
    {
        return (Position[0] << 16) | Get16();
    }
    i32 GetS() const
    {
        return (int(GetU()) << 8) >> 8;
    }
    f32 GetF() const
    {
#if LITTLEENDIAN
        u8 buf[4];
        std::reverse_copy(&AxisStatus, &AxisStatus+4, buf);
        return (f32&)buf[0];
#endif
        return (f32&)AxisStatus;
    }
    struct Status
    {
        u8 Version : 2;
        u8 Summary : 2;
        enum :u8 {
            UNDEFINED = 0,
            TRACKING = 1,
            OPTICAL_GOOD = 2,
            OPTICAL_ACCEPTABLE = 3,
            OPTICAL_UNRELIABLE = 4,
            OPTICAL_UNSTABLE = 5,
            OPTICAL_LOST = 6,
            LOST_TOO_FEW_STARS = 7,
            LOC_SEARCHING = 8,
            BUSY_OR_WAITING = 9,
            BUSY_LOADING_MAP = 10,
            NO_MAP_LOADED = 11,
            TEST_SIGNAL = 12,
            MECH_ENC_ONLY = 13,
            IO_ERROR = 14,
            INTERNAL_ERROR = 15,
        } Detail : 4;
    };

    Status AsStatus() {
        assert(GetType() == STATUS);
        return *(Status*)&Position[1];
    }
};


static_assert(sizeof(MoSysAxis) == 5);
//static_assert(((u8*)&((MoSysAxis*)0)[1] - (u8*)&((MoSysAxis*)0)[0]) == 5);

struct MoSysPacket
{
    u8 Cmd; 
    u8 CameraID; 
    u8 AxisCount;
    u8 Status;
    MoSysAxis Axes[];
    MoSysAxis* begin() { return Axes; }
    MoSysAxis* end() { return Axes + AxisCount; }
};

struct MoSys : public TrackNodeContext
{
    using TrackNodeContext::TrackNodeContext;

    // bool ProcessNextMessage(bool reset, u32 delay, mz::Args &args) override
    bool ProcessNextMessage(std::vector<u8> buf, mz::Args& args)  override
    {
        switch(buf[0])
        {
            case MAGIC_NUMBER:
                break;
            default: 
                return false;
        }
    

        auto packet = ((MoSysPacket*)buf.data());
        
        std::vector<MoSysAxis::XID> ids;

        for (auto axis : *packet) ids.push_back(axis.ID);

        f64 fx = 1920;
        
        for(auto axis : *packet)
        {
            switch(axis.ID)
            {
            case MoSysAxis::TRACKING_STATUS:
            {
                switch (axis.AsStatus().Detail)
                {
                    default:
                    case MoSysAxis::Status::OPTICAL_UNRELIABLE:
                    case MoSysAxis::Status::OPTICAL_UNSTABLE:
                    case MoSysAxis::Status::OPTICAL_LOST:
                    case MoSysAxis::Status::LOST_TOO_FEW_STARS:
                    case MoSysAxis::Status::LOC_SEARCHING:
                    case MoSysAxis::Status::BUSY_OR_WAITING:
                    case MoSysAxis::Status::BUSY_LOADING_MAP:
                    case MoSysAxis::Status::NO_MAP_LOADED:
                    case MoSysAxis::Status::TEST_SIGNAL:
                    case MoSysAxis::Status::MECH_ENC_ONLY:
                    case MoSysAxis::Status::IO_ERROR:
                    case MoSysAxis::Status::INTERNAL_ERROR:
                        return false;
                    case MoSysAxis::Status::UNDEFINED:
                    case MoSysAxis::Status::TRACKING:
                    case MoSysAxis::Status::OPTICAL_GOOD:
                    case MoSysAxis::Status::OPTICAL_ACCEPTABLE:
                        break;
                }
            }
            default: break;
            }

        }

        TrackData.sensor_size.mutate_x(9.590f);
        TrackData.sensor_size.mutate_y(5.394f);
        TrackData.fov = 60.0;
        TrackData.distortion_scale = 1;
        TrackData.pixel_aspect_ratio = 1;

        for(auto axis : *packet)
        {   
            u32 u = axis.GetU();
            i32 s = axis.GetS();
            f64 f = axis.GetF();
            u16 v16 = axis.Get16();
            
            auto ty = axis.GetType();
            if (ty != MoSysAxis::Type::F32 && (axis.AxisStatus & 1))
            {
                continue;
            }
            switch(axis.ID)
            {
                case MoSysAxis::PAN:                TrackData.rotation.mutate_y(s / 1000.); break;
                case MoSysAxis::TILT:               TrackData.rotation.mutate_z(s / 1000.); break;
                case MoSysAxis::ROLL:               TrackData.rotation.mutate_x(s / 1000.); break;
                case MoSysAxis::FOCUS:              TrackData.focus =(v16 / 65535.); break;
                case MoSysAxis::ZOOM:               TrackData.zoom =(v16 / 65535.); break;
                case MoSysAxis::X:                  TrackData.location.mutate_x(s / 1000.); break;
                case MoSysAxis::Y:                  TrackData.location.mutate_y(s / 1000.); break;
                case MoSysAxis::Z:                  TrackData.location.mutate_z(s / 1000.); break;
                case MoSysAxis::CX:                 TrackData.center_shift.mutate_x(f / 1920. * 9.590); break;
                case MoSysAxis::CY:                 TrackData.center_shift.mutate_y(-f / 1080. * 5.394); break;
                case MoSysAxis::LENS_DISTORTION_K1: TrackData.k1k2.mutate_x(f); break;
                case MoSysAxis::LENS_DISTORTION_K2: TrackData.k1k2.mutate_y(f); break;
                case MoSysAxis::FOCAL_LENGTH_FX:    fx = f; break;
                case MoSysAxis::VIRTUAL_FX:         TrackData.fov =(57.2957795131 * 2 * atan(960. / f)); break;
                case MoSysAxis::ENTRANCE_PUPIL:     TrackData.nodal_offset =(f * 100.); break;      
                case MoSysAxis::FOCAL_DISTANCE:
                case MoSysAxis::TIMECODE:
                case MoSysAxis::GENLOCK_INFO:
                default:
                    break;
            }
        }

        const auto c = (960. * 960. + 540. * 540.) / (fx * fx);
        TrackData.k1k2.mutate_x(TrackData.k1k2.x() * c);
        TrackData.k1k2.mutate_x(TrackData.k1k2.y() * c * c);

        UpdateTrackOut(args, *args.GetBuffer("Track"));
        return true;
    }

    ~MoSys()
    {
        if (IsRunning())
        {
            Stop();
        }
    }
};

void RegisterMoSysNode(NodeActionsMap &functions)
{
    auto &actions = functions["mz.MoSys"];
    RegisterTrackCommon(actions);

    actions.NodeCreated = [](fb::Node const &node, mz::Args &args, void **ctx) {
        auto context = new MoSys(args.Get<mz::fb::u16>("UDP_Port")->val());
        *ctx = context;
        auto pins = context->Load(node);
        if (auto pin = pins["UDP_Port"])
        {
            if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
            {
                context->Port = *(uint16_t *)pin->data()->data();
            }
        }
        if (auto pin = pins["Enable"])
        {
            if (flatbuffers::IsFieldPresent(pin, fb::Pin::VT_DATA))
            {
                if (*(bool *)pin->data()->data())
                {
                    context->Start();
                }
            }
        }
    };

    actions.PinValueChanged = [](auto ctx, auto &id, mz::Buffer* value) {
        MoSys *fnctx = (MoSys *)ctx;
        fnctx->OnPinValueChanged(id, value->data());
    };
    actions.NodeRemoved = [](auto ctx, auto &id) { delete (MoSys *)ctx; };
}

} // namespace mz