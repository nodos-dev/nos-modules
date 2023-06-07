/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#include <MediaZ/Helpers.hpp>

#define GLM_FORCE_SWIZZLE 
#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>

#include "mzUtil/Thread.h"
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

MZ_REGISTER_NAME(UDP_Port);
MZ_REGISTER_NAME(Enable);
MZ_REGISTER_NAME(Track);

namespace mz
{
	struct TrackNodeContext : public NodeContext, public Thread
	{
		std::atomic<uint16_t> Port;
		std::atomic_uint Delay;
		std::mutex QMutex;

		std::atomic_uint SpareCount = 0;
		std::atomic_bool YetFillingSpares = true;
		std::atomic_bool ShouldRestart = false;

		std::queue<fb::TTrack> DataQueue;
		std::atomic_uint LastServedFrameNumber = 0;

        struct TransformMapping
        {
            glm::bvec3 NegatePos = {};
            glm::bvec3 NegateRot = {};
            bool EnableEffectiveFOV = true;
            f32 TransformScale = 1.f;
		    mz::fb::CoordinateSystem CoordinateSystem = fb::CoordinateSystem::XYZ;
		    mz::fb::RotationSystem   RotationSystem = fb::RotationSystem::PTR;
            glm::dvec3 DevicePosition = {};
            glm::dvec3 DeviceRotation = {};
            glm::dvec3 CameraPosition = {};
            glm::dvec3 CameraRotation = {};
        };
        
        TransformMapping args;


		TrackNodeContext()
		{
		}

	
        virtual bool Parse(std::vector<u8> const& data, fb::TTrack& out) = 0;

		void OnPathCommand(mz::fb::UUID pinID, app::PathCommand command, Buffer* params)
		{
			switch (command)
			{
			case app::PathCommand::RESTART:
			case app::PathCommand::NOTIFY_DROP:
			{
				ShouldRestart = true;
				mzEngine.LogW("Track queue will be reset", "");
				break;
			}
			}
		}

		bool EntryPoint(mzNodeExecuteArgs const& args)
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
					mzEngine.LogI("Thread active but no data in track queue");
				return false;
			}

			// Get data from queue and resize the fixed-size buffer coming from udp listener thread
// <<<<<<< HEAD
// 			ProcessNextMessage(std::move(DataQueue.front()), args);
			mz::Buffer trackBuf = UpdateTrackOut(DataQueue.front());
			mzEngine.SetPinValueByName(NodeId, MZN_Track, {.Data = trackBuf.data(), .Size = trackBuf.size()});
			DataQueue.pop();
			return true;
        }

		virtual void OnPinValueChanged(mz::fb::UUID const& id, void* value) 
        {
			const auto& pinName = GetPinName(id);

            #define SET_VALUE(ty, name, var) if(pinName ==MZ_NAME_STATIC(#name)) args.##var = *(ty*)value;
            
            SET_VALUE(bool, NegateX, NegatePos.x);
            SET_VALUE(bool, NegateY, NegatePos.y);
            SET_VALUE(bool, NegateZ, NegatePos.z);
            SET_VALUE(bool, NegateRoll, NegateRot.x);
            SET_VALUE(bool, NegatePan, NegateRot.y);
            SET_VALUE(bool, NegateTilt, NegateRot.z);
            SET_VALUE(bool, EnableEffectiveFOV, EnableEffectiveFOV);
            SET_VALUE(f32, TransformScale, TransformScale);
            SET_VALUE(fb::CoordinateSystem, CoordinateSystem, CoordinateSystem);
            SET_VALUE(fb::RotationSystem, Pan/Tilt/Roll, RotationSystem);
            
            SET_VALUE(glm::dvec3, DevicePosition, DevicePosition);
            SET_VALUE(glm::dvec3, DeviceRotation, DeviceRotation);
            SET_VALUE(glm::dvec3, CameraPosition, CameraPosition);
            SET_VALUE(glm::dvec3, CameraRotation, CameraRotation);
			
            if (pinName == MZ_NAME_STATIC("Delay"))
			{
				Delay = *(u32*)value;
				Restart();
				return;
			}

			if (pinName == MZN_Enable)
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
                return;
			}

			if (pinName == MZN_UDP_Port)
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
                return;
			}
			
            if (pinName == MZ_NAME_STATIC("Spare Count"))
			{
				SpareCount = *(uint32_t*)value;
				Restart();
                return;
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

		mz::Buffer UpdateTrackOut(fb::TTrack& outTrack)
		{
			auto xf = args;

			auto pos = Swizzle((glm::dvec3&)outTrack.location, xf.NegatePos, (u8)xf.CoordinateSystem);
			auto rot = Swizzle((glm::dvec3&)outTrack.rotation, xf.NegateRot, (u8)xf.RotationSystem);

			auto CR = MakeRotation(args.CameraRotation);
			auto TR = MakeRotation(rot);
			auto DR = MakeRotation(args.DeviceRotation);

			glm::dvec3 finalPos = DR * (TR * args.CameraPosition + pos) + args.DevicePosition;
			glm::dvec3 finalRot = GetEulers(DR * TR * CR);
			(glm::dvec3&)outTrack.location = finalPos;
			(glm::dvec3&)outTrack.rotation = finalRot;
			
			auto AspectRatio = outTrack.sensor_size.x() / outTrack.sensor_size.y();
			outTrack.distortion_scale = CalculateDistortionScale(AspectRatio, (glm::dvec2&)outTrack.k1k2);

			if (xf.EnableEffectiveFOV)
			{
				outTrack.fov = glm::degrees(2.0f * (atan((outTrack.distortion_scale / 2.0f) * 2.0f * tan(glm::radians(outTrack.fov / 2.0f)))));;
			}

            return mz::Buffer::From(outTrack);
		}

		virtual void Run() override
		{
			flatbuffers::FlatBufferBuilder fbb;
			mzEngine.HandleEvent(
				CreateAppEvent(fbb, mz::app::CreateSetThreadNameDirect(fbb, (u64)StdThread.native_handle(), "Track")));

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
					mzEngine.LogW("could not open UDP socket %d: %s", Port.load(), e.what());

					std::this_thread::sleep_for(std::chrono::seconds(2));
					return;
				}
				u8 buf[4096];
				mzBuffer defaultTrackData;
				mzEngine.GetDefaultValueOfType(MZ_NAME_STATIC("mz.fb.Track"), &defaultTrackData);
				mz::Buffer defaultTrackBuffer = mz::Buffer((uint8_t*)defaultTrackData.Data, defaultTrackData.Size);
				fb::TTrack defaultTrack = defaultTrackBuffer.As<fb::TTrack>();
				while (!ShouldStop)
				{
					try
					{
						udp::endpoint sender_endpoint;
						size_t len = sock->receive_from(asio::buffer(buf, 4096), sender_endpoint);
						{
                            fb::TTrack data = defaultTrack;
							if(Parse(std::vector<u8>{buf, buf + len}, data))
                            {
                                std::unique_lock<std::mutex> guard(QMutex);
                                DataQueue.push(data);

                                // Queue sisiyo mu?
                                // while (DataQueue.size() > Delay) DataQueue.pop();
                                mzEngine.WatchLog("Track Queue Size", std::to_string(DataQueue.size()).c_str());

                                if (YetFillingSpares && DataQueue.size() >= SpareCount)
                                    YetFillingSpares = false;
                            }
						}
					}
					catch (const  asio::system_error& e)
					{
						mzEngine.LogW("Exception when listening on port %d: %s", Port.load(), e.what());
					}
				}
			}

		}
		
	};

	static void RegisterTrackCommon(mzNodeFunctions& functions)
	{
		functions.OnPathCommand = [](void* ctx, const mzPathCommand* command)
		{
			TrackNodeContext* trkCtx = (TrackNodeContext*)ctx;
			mz::Buffer val((u8*)command->Args.Data, command->Args.Size);
			trkCtx->OnPathCommand(command->Id, (app::PathCommand)command->CommandType, &val);
		};

		functions.OnPinConnected = functions.OnPinDisconnected = [](void* ctx, fb::UUID pinId) {
			TrackNodeContext* trkCtx = (TrackNodeContext*)ctx;
			trkCtx->Restart();
		};

		functions.ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) {
			if (((TrackNodeContext*)ctx)->EntryPoint(*args))
				return MZ_RESULT_SUCCESS;
			return MZ_RESULT_FAILED; };
	}
}