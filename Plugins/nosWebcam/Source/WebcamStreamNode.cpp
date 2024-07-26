// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "WebcamStream.h"
#include "nosUtil/Stopwatch.hpp"


NOS_REGISTER_NAME(Device);
NOS_REGISTER_NAME(Format);
NOS_REGISTER_NAME(Resolution);
NOS_REGISTER_NAME(FrameRate);
NOS_REGISTER_NAME(Stream);
namespace nos::webcam
{
enum class ChangedPinType
{
	Device,
	FormatName,
	Resolution,
	FrameRate
};

struct WebcamStreamNode : public nos::NodeContext
{
	WebcamStreamNode(const nosFbNode* node) : nos::NodeContext(node)
	{
		DeviceList = WebcamStreamManager::EnumerateDevices();

		UpdateStringList(GetDeviceStringListName(), GetDeviceList());
		UpdateStringList(GetFormatStringListName(), { "NONE" });
		UpdateStringList(GetResolutionStringListName(), { "NONE" });
		UpdateStringList(GetFrameRateStringListName(), { "NONE" });

		SetPinVisualizer(NSN_Device, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetDeviceStringListName() });
		SetPinVisualizer(NSN_Format, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetFormatStringListName() });
		SetPinVisualizer(NSN_Resolution, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetResolutionStringListName() });
		SetPinVisualizer(NSN_FrameRate, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetFrameRateStringListName() });

		AddPinValueWatcher(NSN_Device, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				DevicePin = InterpretPinValue<char>(newVal);
				SelectedDevice = std::nullopt;
				CurDeviceFormats.clear();
				if (DevicePin != "NONE")
				{
					for (auto const& device : DeviceList)
						if (device.Name == DevicePin)
						{
							SelectedDevice = device;
							CurDeviceFormats = WebcamStreamManager::EnumerateFormats(*SelectedDevice);
							break;
						}
				}
				if (!SelectedDevice && !oldValue)
				{
					if (!DeviceList.empty())
					{
						AutoSelectIfPossible(NSN_Device, GetDeviceList());
						return;
					}
					else if(DevicePin != "NONE")
					{
						SetPinValue(NSN_Device, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(ChangedPinType::Device, !oldValue);
			});

		AddPinValueWatcher(NSN_Format, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				FormatPin = InterpretPinValue<char>(newVal);
				SelectedFormatGuid = std::nullopt;
				if (FormatPin != "NONE")
				{
					SelectedFormatGuid = GetSubTypeFromFormatName(FormatPin);
				}
				if (!SelectedFormatGuid)
				{
					if (auto formatList = GetFormatList(); formatList.size() > 1)
					{
						AutoSelectIfPossible(NSN_Format, formatList);
						return;
					}
					else if(FormatPin != "NONE")
					{
						SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(ChangedPinType::FormatName, !oldValue);
			});

		AddPinValueWatcher(NSN_Resolution, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				ResolutionPin = InterpretPinValue<char>(newVal);
				SelectedResolution = std::nullopt;
				if (ResolutionPin != "NONE")
				{
					SelectedResolution = GetResolutionFromString(ResolutionPin);
				}
				if (!SelectedResolution)
				{
					if (auto resolutionList = GetResolutionList(); resolutionList.size() > 1)
					{
						AutoSelectIfPossible(NSN_Resolution, resolutionList);
						return;
					}
					else if (ResolutionPin != "NONE")
					{
						SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(ChangedPinType::Resolution, !oldValue);
			});

		AddPinValueWatcher(NSN_FrameRate, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				FrameRatePin = InterpretPinValue<char>(newVal);
				SelectedFrameRate = std::nullopt;
				if (FrameRatePin != "NONE")
				{
					SelectedFrameRate = GetFrameRateFromString(FrameRatePin);
				}
				if (!SelectedFrameRate)
				{
					if (auto frameRateList = GetFrameRateList(); frameRateList.size() > 1)
					{
						AutoSelectIfPossible(NSN_FrameRate, frameRateList);
						return;
					}
					else if(FrameRatePin != "NONE")
					{
						SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(ChangedPinType::FrameRate, !oldValue);
			});
	}

	~WebcamStreamNode()
	{
		CloseStream();
	}

	bool TryOpenDevice()
	{
		CloseStream();
		if (!SelectedDevice || !SelectedFormatGuid || !SelectedResolution || !SelectedFrameRate)
			return false;
		FormatInfo info{};
		info.SubType = *SelectedFormatGuid;
		info.Resolution = *SelectedResolution;
		info.FrameRate = *SelectedFrameRate;
		bool found = false;
		for (auto& format : CurDeviceFormats)
		{
			if (info.SubType == format.SubType && info.FrameRate == format.FrameRate && info.Resolution == format.Resolution)
			{
				SelectedFormatInfo = format;
				found = true;
				break;
			}
		}
		if (!found)
			return false;
			
		if (auto res = WebcamStreamManager::GetInstance().OpenStreamFromFormat(*SelectedDevice, SelectedFormatInfo); res.has_value())
		{
			auto openedStream = res.value();
			StreamId = openedStream->StreamId;
			SetPinValue(NSN_Stream, nos::Buffer::From(openedStream->GetStreamInfo()));
			nosEngine.SendPathRestart(NodeId);
			return true;
		}
		else
		{
			nosEngine.LogE("Failed to open webcam stream: %s", res.error().c_str());
			return false;
		}
	}

	void CloseStream()
	{
		nosEngine.SendPathRestart(NodeId);
		SelectedFormatInfo = {};
		if(StreamId)
			WebcamStreamManager::GetInstance().DeleteStream(*StreamId);
		SetPinValue(NSN_Stream, nos::Buffer::From(TWebcamStreamInfo{}));
	}

	void UpdateAfter(ChangedPinType type, bool first)
	{
		switch (type)
		{
		case ChangedPinType::Device:
		{
			auto formatList = GetFormatList();
			UpdateStringList(GetFormatStringListName(), formatList);
			if (!SelectedDevice)
				SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else if(!first)
				AutoSelectIfPossible(NSN_Format, formatList);
			break;
		}
		case ChangedPinType::FormatName:
		{
			auto resolutionList = GetResolutionList();
			UpdateStringList(GetResolutionStringListName(), resolutionList);
			if (!SelectedFormatGuid)
				SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else if(!first)
				AutoSelectIfPossible(NSN_Resolution, resolutionList);

			break;
		}
		case ChangedPinType::Resolution:
		{
			auto frameRateList = GetFrameRateList();
			UpdateStringList(GetFrameRateStringListName(), frameRateList);
			if (!SelectedResolution)
				SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else if(!first)
				AutoSelectIfPossible(NSN_FrameRate, frameRateList);
			break;
		}
		case ChangedPinType::FrameRate:
		{
			if (SelectedFrameRate)
				TryOpenDevice();
			else
				CloseStream();
			break;
		}
		}
	}

	std::vector<std::string> GetDeviceList()
	{
		std::vector<std::string> ret = { "NONE" };
		for (auto const& [name, id] : DeviceList)
			ret.push_back(name);
		return ret;
	}

	std::vector<std::string> GetFormatList()
	{
		std::vector<std::string> ret = { "NONE" };
		if (!SelectedDevice)
			return ret;
		for (auto const& format : CurDeviceFormats)
		{
			if (auto formatStr = GetFormatNameFromSubType(format.SubType); std::find(ret.begin(), ret.end(), formatStr) == ret.end())
				ret.push_back(formatStr);
		}
		return ret;
	}

	std::vector<std::string> GetResolutionList()
	{
		std::vector<std::string> ret = { "NONE" };
		if (!SelectedDevice || !SelectedFormatGuid)
			return ret;
		for (auto const& format : CurDeviceFormats)
			if (format.SubType == *SelectedFormatGuid)
				if (auto resStr = GetResolutionString(format.Resolution); std::find(ret.begin(), ret.end(), resStr) == ret.end())
					ret.push_back(resStr);
		return ret;
	}

	std::vector<std::string> GetFrameRateList()
	{
		std::vector<std::string> ret = { "NONE" };
		if (!SelectedDevice || !SelectedFormatGuid || !SelectedResolution)
			return ret;
		for (auto const& format : CurDeviceFormats)
			if (format.SubType == *SelectedFormatGuid && format.Resolution == *SelectedResolution)
				if (auto frameRateStr = GetFrameRateString(format.FrameRate); std::find(ret.begin(), ret.end(), frameRateStr) == ret.end())
					ret.push_back(frameRateStr);
		return ret;
	}

	std::string GetDeviceStringListName() { return "webcam.DeviceList." + UUID2STR(NodeId); }
	std::string GetFormatStringListName() { return "webcam.FormatList." + UUID2STR(NodeId); }
	std::string GetResolutionStringListName() { return "webcam.ResolutionList." + UUID2STR(NodeId); }
	std::string GetFrameRateStringListName() { return "webcam.FrameRateList." + UUID2STR(NodeId); }

	void AutoSelectIfPossible(nosName pinName, std::vector<std::string> const& list)
	{
		assert(list.size() > 1);
		SetPinValue(pinName, nosBuffer{ .Data = (void*)list[1].c_str(), .Size = list[1].size() + 1 });
	}

	std::optional<nosUUID> StreamId;
	FormatInfo SelectedFormatInfo;
	int WebCamIndex = 0;

	std::optional<WebcamDevice> SelectedDevice;
	std::optional<GUID> SelectedFormatGuid;
	std::optional<nos::fb::vec2u> SelectedResolution;
	std::optional<WebcamFrameRate> SelectedFrameRate;

	std::string DevicePin = "NONE";
	std::string FormatPin = "NONE";
	std::string ResolutionPin = "NONE";
	std::string FrameRatePin = "NONE";

	nosResourceShareInfo _nosIntermediateTexture = {};
	nosResourceShareInfo _nosMemoryBuffer = {};
	std::vector<WebcamDevice> DeviceList;
	std::vector<FormatInfo> CurDeviceFormats;
};
nosResult RegisterWebcamStream(nosNodeFunctions* outFunc)
{
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.webcam.WebcamStream"), nos::webcam::WebcamStreamNode, outFunc);
	return NOS_RESULT_SUCCESS;
}
}; // namespace nos::webcam
