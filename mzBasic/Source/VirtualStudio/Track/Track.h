/*
 * Copyright MediaZ AS. All Rights Reserved.
 */


#define GLM_FORCE_SWIZZLE 
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>

#include "BasicMain.h"

#include <mzFlatBuffersCommon.h>
#include <Args.h>
#include <Builtins_generated.h>

#include "mzUtil/Thread.h"
#include <mzFlatBuffersCommon.h>
#include <AppService_generated.h>

#include <asio.hpp>
#include <atomic>
#include <cmath>


using asio::ip::udp;
typedef uint8_t uint8;
typedef int8_t int8;
typedef uint32_t uint32;
typedef int32_t int32;

static glm::dmat3 MakeRotation(glm::dvec3 rot)
{
    rot = glm::radians(rot);
    return (glm::mat3)glm::eulerAngleZYX(rot.z, -rot.y, -rot.x);
}

static glm::dvec3 GetEulers(glm::dmat4 mat)
{
    f64 x,y,z;
    glm::extractEulerAngleZYX(mat, z, y, x);
    return glm::degrees(glm::dvec3(-x, -y, z));
}


namespace mz
{
	struct TrackNodeContext : public NodeContext, public Thread
	{
		mz::EngineNodeServices NodeServices;
		std::atomic<uint16_t> Port;
		std::atomic_uint Delay;
		std::mutex QMutex;
		std::queue<std::vector<u8>> DataQueue;
		fb::TTrack TrackData;
		std::atomic_uint SpareCount = 0;
		std::atomic_bool YetFillingSpares = true;
		std::atomic_bool ShouldRestart = false;

        struct TransformMapping
        {
            glm::bvec3 NegatePos;
            glm::bvec3 NegateRot;
            bool EnableEffectiveFov;
            f32 TransformScale = 1.f;
		    mz::fb::CoordinateSystem CoordinateSystem;
		    mz::fb::RotationSystem   RotationSystem;
        };
        
        static TransformMapping GetXfMapping(mz::Args& args)
        {
            return {
                .NegatePos = { 
                    args.GetOrDefo<bool>("NegateX"),
                    args.GetOrDefo<bool>("NegateY"),
                    args.GetOrDefo<bool>("NegateZ"),
                },
                .NegateRot = { 
                    args.GetOrDefo<bool>("NegateRoll"),
                    args.GetOrDefo<bool>("NegatePan"),
                    args.GetOrDefo<bool>("NegateTilt"),
                },
                .EnableEffectiveFov = args.GetOrDefo<bool>("EnableEffectiveFOV"),
                .TransformScale = args.GetOrDefo<f32>("TransformScale", 1.f),
                .CoordinateSystem = args.GetOrDefo<mz::fb::CoordinateSystem>("CoordinateSystem", fb::CoordinateSystem::XYZ),
                .RotationSystem = args.GetOrDefo<mz::fb::RotationSystem>("Pan/Tilt/Roll", fb::RotationSystem::PTR),
            };
        }

		TrackNodeContext(uint16_t port) :
			Port(port),
			NodeServices(GServices),
            TrackData()
		{
            if(auto def = GServices.GetDefaultDataOfType("mz.fb.Track"))
            {
                TrackData = def->As<fb::TTrack>();
            }
		}

		virtual bool ProcessNextMessage(std::vector<u8> data, mz::Args& args) = 0;

		void OnPathCommand(mz::fb::UUID pinID, app::PathCommand command, Buffer* params)
		{
			switch (command)
			{
			case app::PathCommand::RESTART:
			case app::PathCommand::NOTIFY_DROP:
			{
				ShouldRestart = true;
				GServices.LogW("Track queue will be reset", "");
				break;
			}
			}
		}

		bool EntryPoint(mz::Args& args)
		{
			if (ShouldRestart)
			{
				Restart();
				ShouldRestart = false;
			}

			if (YetFillingSpares && DataQueue.size() < SpareCount)
				return false;

            std::unique_lock<std::mutex> guard(QMutex);
			if (DataQueue.size() <= Delay)
			{
				if (IsRunning())
					GServices.Log("Thread active but no data in track queue", "");
				return false;
			}

			// Get data from queue and resize the fixed-size buffer coming from udp listener thread
			ProcessNextMessage(std::move(DataQueue.front()), args);
			DataQueue.pop();
			return true;
        }

		virtual void OnPinValueChanged(mz::fb::UUID const& id, void* value) 
        {
			const auto& pinName = GetPinName(id);
			if (pinName == "Delay")
			{
				Delay = *(u32*)value;
				Restart();
				return;
			}
			else if (pinName == "Enable")
			{
				Restart();
				auto enable = *(bool*)value;
				if (enable)
				{
					if (!IsRunning())
					{
						Join();
						Start();
					}
				}
				else
				{
					if (IsRunning())
					{
						Stop();
					}
				}
			}
			else if (pinName == "UDP_Port")
			{
				if (IsRunning())
				{
					Stop();
					auto newPort = *(uint16_t*)value;
					Port = newPort;
					Start();
				}
				else
				{
					auto newPort = *(uint16_t*)value;
					Port = newPort;
				}
			}
			else if (pinName == "Spare Count")
			{
				SpareCount = *(uint32_t*)value;
				Restart();
			}
        }

		void Restart(u32 spareCount = 0)
		{
			std::unique_lock<std::mutex> guard(QMutex);
			while (DataQueue.size() > spareCount)
				DataQueue.pop();
			YetFillingSpares = true;
		}

		f64 CalculateR(f64 R, glm::dvec2 k1k2)
		{
			f64 R2 = R * R;
			f64 R4 = R2 * R2;
			return k1k2.x * R2 + k1k2.y * R4 + 1;
		}

		f64 CalculateRoot(f64 TargetR, glm::dvec2 k1k2, f64 InitialR)
		{
			f64 R = InitialR;
			for (int t = 0; t < 10; ++t)
			{
				f64 R2 = R * R;
				f64 R3 = R2 * R;
				f64 R4 = R2 * R2;
				f64 R5 = R3 * R2;
				f64 fR = k1k2.x * R3 + k1k2.y * R5 + R - TargetR; // (K1 * R2 + K2 * R4 + 1) * R - TargetR;
				f64 dfR = 3 * k1k2.x * R2 + 5 * k1k2.y * R4 + 1;
				f64 hR = fR / dfR;
				R = R - hR;
			}
			return R;
		}

		f64 CalculateDistortionScale(f64 AspectRatio, glm::dvec2 k1k2)
		{
			auto AspectVector = glm::dvec2(AspectRatio, 1);
			f64 X = sqrt(1.0 / (AspectVector.x * AspectVector.x + AspectVector.y * AspectVector.y));
			auto AspectRatioVector = AspectVector * X;
			glm::dvec2 P = glm::dvec2(0., 1.) * AspectRatioVector;
			f64 PLength = glm::length(P);
			f64 YMin = CalculateRoot(PLength, k1k2, 1.0) / PLength;
			// float YMax = YMin;
			glm::dvec2 PMin = P;
			// FVector2D PMax = P;	
			int32 IterCount = 1000;
			f64 IterStep = 1. / IterCount;
			for (int32 Iter = 0; Iter < IterCount; ++Iter)
			{
				P = glm::dvec2(Iter * IterStep, 1.0) * AspectRatioVector;
				PLength = glm::length(P);
				float Y = CalculateRoot(PLength, k1k2, 1.0f) / PLength;
				if (Y < YMin)
				{
					YMin = Y;
					PMin = P;
				}
				/*
				if (Y > YMax)
				{
					YMax = Y;
					PMax = P;
				}
				*/
			}
			auto PMinLength = glm::length(PMin);
			// auto PMaxLength = PMax.Size();
			auto ScaleMinRoot = CalculateRoot(PMinLength, k1k2, 1.0);
			// auto ScaleMaxRoot = CalculateRoot(PMaxLength, K1, K2, 1.0f);		
			auto ScaleMin = ScaleMinRoot / PMinLength;
			// auto ScaleMax = ScaleMaxRoot / PMaxLength;	
			auto SMin = CalculateR(PMinLength, k1k2);
			// auto SMax = CalculateR(PMaxLength, K1, K2);
			return SMin;
		}

        static glm::dvec3 Swizzle(glm::dvec3 v, glm::bvec3 n, u8 control)
        {
            if(control & 0b001) v = v.zyx;
            if(control & 0b010) v = v.yzx;
            if(control & 0b100) v = v.zxy;
            return glm::mix(v, -v, n);
        }

		void UpdateTrackOut(mz::Args& args, mz::Buffer& out)
		{
			auto outTrack = TrackData;
			auto xf = GetXfMapping(args);

			auto pos = Swizzle((glm::dvec3&)TrackData.location, xf.NegatePos, (u8)xf.CoordinateSystem);
			auto rot = Swizzle((glm::dvec3&)TrackData.rotation, xf.NegateRot, (u8)xf.RotationSystem);

			auto dpos = *args.Get<glm::dvec3>("DevicePosition");
			auto drot = *args.Get<glm::dvec3>("DeviceRotation");
			auto cpos = *args.Get<glm::dvec3>("CameraPosition");
			auto crot = *args.Get<glm::dvec3>("CameraRotation");

			auto CR = MakeRotation(crot);
			auto TR = MakeRotation(rot);
			auto DR = MakeRotation(drot);

			glm::dvec3 finalPos = DR * (TR * cpos + pos) + dpos;
			glm::dvec3 finalRot = GetEulers(DR * TR * CR);
			(glm::dvec3&)outTrack.location = finalPos;
			(glm::dvec3&)outTrack.rotation = finalRot;
			
			auto AspectRatio = TrackData.sensor_size.x() / TrackData.sensor_size.y();
			outTrack.distortion_scale = CalculateDistortionScale(AspectRatio, (glm::dvec2&)TrackData.k1k2);

			if (xf.EnableEffectiveFov)
			{
				outTrack.fov = glm::degrees(2.0f * (atan((TrackData.distortion_scale / 2.0f) * 2.0f * tan(glm::radians(TrackData.fov / 2.0f)))));;
			}

			out = mz::Buffer::From(outTrack);
			// auto outTrack = out.As<mz::fb::Track>();
		}

		virtual void Run() override
		{
			flatbuffers::FlatBufferBuilder fbb;
			GServices.HandleEvent(
				CreateAppEvent(fbb, mz::app::CreateSetThreadNameDirect(fbb, (u64)m_Thread.native_handle(), "Track")));

			asio::io_service io_serv;
			{
				rc<udp::socket> sock;
				try
				{
					sock = MakeShared<udp::socket>(io_serv, udp::endpoint(udp::v4(), Port));
					sock->set_option(asio::detail::socket_option::integer<SOL_SOCKET, SO_RCVTIMEO>{1000});
				}
				catch (const  asio::system_error& e)
				{
					GServices.LogW("could not open UDP socket" + std::to_string(Port), e.what());

					std::this_thread::sleep_for(std::chrono::seconds(2));
					return;
				}
				u8 buf[4096];

				while (!m_ShouldStop)
				{
					try
					{
						udp::endpoint sender_endpoint;
						size_t len = sock->receive_from(asio::buffer(buf, 4096), sender_endpoint);
						{
							std::unique_lock<std::mutex> guard(QMutex);
							DataQueue.push(std::vector<u8>{buf, buf + len});
							GServices.LogFast("Track Queue Size", std::to_string(DataQueue.size()));

							if (YetFillingSpares && DataQueue.size() >= SpareCount)
								YetFillingSpares = false;
						}
					}
					catch (const  asio::system_error& e)
					{
						GServices.LogW("timeout when listening on port " + std::to_string(Port), e.what());
					}
				}
			}

		}
		
	};

	static void RegisterTrackCommon(NodeActions& actions)
	{
		actions.OnPathCommand = [](fb::UUID pinID, app::PathCommand command, Buffer* param, void* ctx)
		{
			TrackNodeContext* trkCtx = (TrackNodeContext*)ctx;
			trkCtx->OnPathCommand(pinID, command, param);
		};

		actions.PinConnected = actions.PinDisconnected = [](void* ctx, fb::UUID const& pinId) {
			TrackNodeContext* trkCtx = (TrackNodeContext*)ctx;
			trkCtx->Restart();
		};

		actions.EntryPoint = [](mz::Args& args, void* ctx) {  return ((TrackNodeContext*)ctx)->EntryPoint(args); };
	}
}