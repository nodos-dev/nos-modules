// Copyright MediaZ AS. All Rights Reserved.


#include "Track.h"

#include "BasicMain.h"

#include <mzFlatBuffersCommon.h>
#include <Args.h>
#include <Builtins_generated.h>


#include "flatbuffers/flatbuffer_builder.h"
#include "mzUtil/Thread.h"
#include <mzFlatBuffersCommon.h>
#include <AppService_generated.h>

#include <asio.hpp>
#include <atomic>


using asio::ip::udp;
typedef uint8_t uint8;
typedef int8_t int8;
typedef uint32_t uint32;
typedef int32_t int32;

namespace mz
{

	struct FZDFreeDFloat
	{
		uint8 Value1;

		uint8 Value2;

		uint8 Value3;

		FZDFreeDFloat()
			: Value1(0), Value2(0), Value3(0)
		{
		}

		float GetValue() const
		{
			return (Value1 << 16) | (Value2 << 8) | (Value3);
		}
	};

	struct FZDFreeDFloat_Rotation
	{
		uint8 Value1;

		uint8 Value2;

		uint8 Value3;

		FZDFreeDFloat_Rotation()
			: Value1(0), Value2(0), Value3(0)
		{
		}

		float GetValue(const float& Constant) const
		{
			uint8 PreValue = ((Value1 & 0x80) == 0x80) ? 0xff : 0x00;
			uint32 Value = (PreValue << 24) | (Value1 << 16) | (Value2 << 8) | (Value3);
			return (*(int32*)(&Value)) / Constant;
		}
	};

	struct FZDFreeDFloat_Location
	{
		uint8 Value1;

		uint8 Value2;

		uint8 Value3;

		FZDFreeDFloat_Location()
			: Value1(0), Value2(0), Value3(0)
		{
		}

		float GetValue(const float& Constant) const
		{
			uint8 PreValue = ((Value1 & 0x80) == 0x80) ? 0xff : 0x00;
			uint32 Value = (PreValue << 24) | (Value1 << 16) | (Value2 << 8) | (Value3);
			return (*(int32*)(&Value)) / Constant;
		}
	};



	struct FZDFreeDIntegerAndFraction
	{
		int8 Value1;

		uint8 Value2;

		int8 Value3;

		uint8 Value4;

		FZDFreeDIntegerAndFraction()
			: Value1(0), Value2(0), Value3(0), Value4(0)
		{
		}

		int32 GetFractionValue() const
		{
			int8 PreValue = ((Value1 & 0x00) == 0x00) ? 0xff : 0x00;
			uint32 Value = (PreValue << 16) | (Value1 << 8) | (Value2);
			return (*(int32*)(&Value));
		}

		int32 GetIntegerValue() const
		{
			int8 PreValue = ((Value3 & 0x80) == 0x80) ? 0xff : 0x00;
			int32 Value = (PreValue << 16) | (Value3 << 8) | (Value4);
			return (*(int32*)(&Value));
		}

		// TODO: needs professional eyes.
		float GetValue(const float& Constant) const
		{
			auto IntegerValue = GetIntegerValue();
			auto FractionValue = GetFractionValue();
			float midvalue = 0.0f;
			if (IntegerValue >= 0)
			{
				midvalue = (float)IntegerValue + ((float)FractionValue / UINT16_MAX) + 1.0f;
			}
			else
			{
				midvalue = (float)IntegerValue - ((float)FractionValue / UINT16_MAX) - 1.0f;
			}

			return midvalue / Constant;
		}
	};

	struct FZDFreeDInteger
	{
		uint8 Value1;

		uint8 Value2;

		uint8 Value3;

		FZDFreeDInteger()
			: Value1(0), Value2(0), Value3()
		{
		}

		uint32 GetIntegerValue() const
		{
			return (Value1 << 16) | (Value2 << 8) | (Value3);
		}

		// Zoom ve focus icin test edilmedi.
		float GetValue(const float& MinConstant, const float& MaxConstant) const
		{
			auto IntegerValue = GetIntegerValue();
			return (float)(IntegerValue - MinConstant) / (MaxConstant - MinConstant);
		}
	};


	struct FZDFreeDMessage_D1
	{
		uint8 Header;

		uint8 CameraID;

		FZDFreeDFloat_Rotation Pan;

		FZDFreeDFloat_Rotation Tilt;

		FZDFreeDFloat_Rotation Roll;

		FZDFreeDFloat_Location X;

		FZDFreeDFloat_Location Y;

		FZDFreeDFloat_Location Z;

		FZDFreeDInteger Zoom;

		FZDFreeDInteger Focus;

		uint8 SpareData1;

		uint8 SpareData2;

		uint8 Checksum;

		glm::vec3 GetLocation(const glm::vec3& Constants) const
		{
			return glm::vec3(X.GetValue(Constants.x), Y.GetValue(Constants.y), Z.GetValue(Constants.z));
		}

		glm::vec3 GetRotation(const glm::vec3& Constants) const
		{
			return glm::vec3(Tilt.GetValue(Constants.x), Pan.GetValue(Constants.y), Roll.GetValue(Constants.z));
		}

		bool IsChecksumOK() const
		{
			//The checksum is calculated by subtracting (modulo 256) each byte 
			//of the message, including the messageype, from 40 (hex).
			const uint8* Buffer = (const uint8*)this;
			uint8 TotalSum = 0x40;
			for (uint32 Index = 0; Index < sizeof(FZDFreeDMessage_D1) - 1; ++Index)
			{
				TotalSum -= Buffer[Index];
			}
			if (TotalSum != Checksum)
			{
				return false;
			}
			return true;
		}

	};

struct FreeDNodeContext : public TrackNodeContext
{
	public:
		FreeDNodeContext(uint16_t port) :
			TrackNodeContext(port)
		{
		}
		
        bool ProcessNextMessage(std::vector<u8> data, mz::Args& args)  override
		{
			FZDFreeDMessage_D1 d1msg = *(FZDFreeDMessage_D1*)data.data();
			auto Location = (glm::dvec3)d1msg.GetLocation(glm::vec3(640));
			auto Rotation = (glm::dvec3)d1msg.GetRotation(glm::vec3(32768));
			auto Zoom = d1msg.Zoom.GetValue(0.0f, 60000.0f);
			auto Focus = d1msg.Focus.GetValue(0.0f, 60000.0f);


			TrackData.zoom = (Zoom);
			TrackData.focus = (Focus);
			TrackData.location = (mz::fb::vec3d&)Location;
			TrackData.rotation = (mz::fb::vec3d&)Rotation;

			UpdateTrackOut(args, *args.GetBuffer("Track"));
			return true;
		}

		~FreeDNodeContext() {
			if (IsRunning())
			{
				Stop();
			}
		}
};

void RegisterFreeDNode(NodeActionsMap& functions)
{
	auto& actions = functions["mz.FreeD"];

	RegisterTrackCommon(actions);

	actions.NodeCreated = [](fb::Node const& node, mz::Args& args, void** ctx) {
		auto context = new FreeDNodeContext(args.Get<mz::fb::u16>("UDP_Port")->val());
		*ctx = context;
		auto pins = context->Load(node);
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

	actions.PinValueChanged = [](auto ctx, auto& id, mz::Buffer* value) 
	{ 
		FreeDNodeContext* fnctx = (FreeDNodeContext*)ctx;
		fnctx->OnPinValueChanged(id, value->data());
	};
	actions.NodeRemoved = [](auto ctx, auto& id) {
		delete (FreeDNodeContext*)ctx;
	};
}

} // namespace mz