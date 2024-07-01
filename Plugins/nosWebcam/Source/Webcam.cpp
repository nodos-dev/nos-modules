// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "MFCapture.h"
#include "nosUtil/Stopwatch.hpp"

NOS_INIT_WITH_MIN_REQUIRED_MINOR(0)
NOS_VULKAN_INIT();

namespace nos::webcam
{

NOS_REGISTER_NAME(Device);
NOS_REGISTER_NAME(Format);
NOS_REGISTER_NAME(Resolution);
NOS_REGISTER_NAME(FrameRate);
NOS_REGISTER_NAME(OutResolution);

enum class WebcamChangedPinType
{
	Device,
	FormatName,
	Resolution,
	FrameRate
};

struct WebcamInNode : public nos::NodeContext
{
	WebcamInNode(const nosFbNode* node) : nos::NodeContext(node), Cap()
	{
		DeviceList = Cap.EnumerateDevices();

		UpdateStringList(GetDeviceStringListName(), GetDeviceList());
		UpdateStringList(GetFormatStringListName(), { "NONE" });
		UpdateStringList(GetResolutionStringListName(), { "NONE" });
		UpdateStringList(GetFrameRateStringListName(), { "NONE" });

		SetPinVisualizer(NSN_Device, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetDeviceStringListName() });
		SetPinVisualizer(NSN_Format, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetFormatStringListName() });
		SetPinVisualizer(NSN_Resolution, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetResolutionStringListName() });
		SetPinVisualizer(NSN_FrameRate, { .type = nos::fb::VisualizerType::COMBO_BOX, .name = GetFrameRateStringListName() });
	}

	bool TryOpenDevice()
	{
		CloseStream();
		if (!SelectedDeviceId || !SelectedFormatGuid || !SelectedResolution || !SelectedFrameRate)
			return false;
		FormatInfo info{};
		info.SubType = *SelectedFormatGuid;
		info.Resolution = *SelectedResolution;
		info.FrameRate = *SelectedFrameRate;
		bool found = false;
		for (auto& format : CurDeviceFormats)
		{
			if(info.SubType == format.SubType && info.FrameRate == format.FrameRate && info.Resolution == format.Resolution)
			{
				SelectedFormatInfo = format;
				found = true;
				break;
			}
		}
		if (!found)
			return false;
		found = false;
		if (auto err = Cap.CreateStreamFromFormat(*SelectedDeviceId, SelectedFormatInfo))
		{
			nosEngine.LogE("Failed to open webcam stream: %s", err->c_str());
			CloseStream();
			return false;
		}
		SetPinValue(NSN_OutResolution, nos::Buffer::From(SelectedFormatInfo.Resolution));
		nosEngine.SendPathRestart(NodeId);
		return true;
	}

	void CloseStream()
	{
		nosEngine.SendPathRestart(NodeId);
		SelectedFormatInfo = {};
		Cap.CloseStream();
		SetPinValue(NSN_OutResolution, nos::Buffer::From(nos::fb::vec2u()));
	}

	void UpdateAfter(WebcamChangedPinType type, bool first)
	{
		switch (type)
		{
		case WebcamChangedPinType::Device:
		{
			SelectedDeviceId = std::nullopt;
			if (DevicePin == "NONE")
				break;
			for (auto const& [name, id] : DeviceList)
				if (name == DevicePin)
				{
					SelectedDeviceId = id;
					break;
				}
			CurDeviceFormats.clear();
			if (SelectedDeviceId)
				CurDeviceFormats = Cap.EnumerateFormats(*SelectedDeviceId);
			else if (DevicePin != "NONE")
				SetPinValue(NSN_Device, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			UpdateStringList(GetFormatStringListName(), GetFormatList());
			if(!SelectedDeviceId || !first)
				SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			break;
		}
		case WebcamChangedPinType::FormatName:
		{
			SelectedFormatGuid = GetSubTypeFromFormatName(FormatPin);
			if(!SelectedFormatGuid && FormatPin != "NONE")
				SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			UpdateStringList(GetResolutionStringListName(), GetResolutionList());
			if(!SelectedFormatGuid || !first)
			SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			break;
		}
		case WebcamChangedPinType::Resolution:
		{
			SelectedResolution = GetResolutionFromString(ResolutionPin);
			if (!SelectedResolution && ResolutionPin != "NONE")
				SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			UpdateStringList(GetFrameRateStringListName(), GetFrameRateList());
			if (!SelectedResolution || !first)
				SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			break;
		}
		case WebcamChangedPinType::FrameRate:
		{
			SelectedFrameRate = GetFrameRateFromString(FrameRatePin);
			if (SelectedFrameRate)
				TryOpenDevice();
			else
			{
				CloseStream();
				if (FrameRatePin != "NONE")
					SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			}
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
		if (!SelectedDeviceId)
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
		if (!SelectedDeviceId || !SelectedFormatGuid)
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
		if (!SelectedDeviceId || !SelectedFormatGuid || !SelectedResolution)
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

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (pinName == NSN_Device)
		{
			DevicePin = InterpretPinValue<char>(value);
			UpdateAfter(WebcamChangedPinType::Device, false);
			UpdateStringList(GetFormatStringListName(), GetFormatList());
		}
		else if (pinName == NSN_Format)
		{
			FormatPin = InterpretPinValue<char>(value);
			UpdateAfter(WebcamChangedPinType::FormatName, false);
			UpdateStringList(GetResolutionStringListName(), GetResolutionList());
		}
		else if (pinName == NSN_Resolution)
		{
			ResolutionPin = InterpretPinValue<char>(value);
			UpdateAfter(WebcamChangedPinType::Resolution, false);
			UpdateStringList(GetFrameRateStringListName(), GetFrameRateList());
		}
		else if (pinName == NSN_FrameRate)
		{
			FrameRatePin = InterpretPinValue<char>(value);
			UpdateAfter(WebcamChangedPinType::FrameRate, false);
		}
	}


	// Execution
	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args)
	{
		if (SelectedFormatInfo.Resolution.x() == 0)
			return NOS_RESULT_FAILED;

		auto sample = Cap.ReadSample();

		// First sample is not valid, try again
		if (sample.Size == 0)
		{
			sample = Cap.ReadSample();
			if (sample.Size == 0)
				return NOS_RESULT_FAILED;
		}

		nos::NodeExecuteArgs execArgs(args);

		auto buffer = vkss::ConvertToResourceInfo(*InterpretPinValue<nos::sys::vulkan::Buffer>(*execArgs[NOS_NAME("Output")].Data));

		if (buffer.Info.Buffer.Size != sample.Size)
		{
			nosResourceShareInfo bufInfo{};
			bufInfo.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
			bufInfo.Info.Buffer.Size = sample.Size;
			bufInfo.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_STORAGE_BUFFER);
			bufInfo.Info.Buffer.MemoryFlags = nosMemoryFlags(NOS_MEMORY_FLAGS_HOST_VISIBLE);

			nosEngine.SetPinValue(execArgs[NOS_NAME("Output")].Id, nos::Buffer::From(vkss::ConvertBufferInfo(bufInfo)));
			buffer = vkss::ConvertToResourceInfo(*InterpretPinValue<nos::sys::vulkan::Buffer>(*execArgs[NOS_NAME("Output")].Data));
		}

		uint8_t* mapped = nosVulkan->Map(&buffer);

		memcpy(mapped, sample.Data, sample.Size);


		return NOS_RESULT_SUCCESS;
	}

	Capturer Cap;
	FormatInfo SelectedFormatInfo;
	int WebCamIndex = 0;

	std::optional<std::wstring> SelectedDeviceId;
	std::optional<GUID> SelectedFormatGuid;
	std::optional<nos::fb::vec2u> SelectedResolution;
	std::optional<nos::fb::vec2u > SelectedFrameRate;

	std::string DevicePin = "NONE";
	std::string FormatPin = "NONE";
	std::string ResolutionPin = "NONE";
	std::string FrameRatePin = "NONE";

	nosResourceShareInfo _nosIntermediateTexture = {};
	nosResourceShareInfo _nosMemoryBuffer = {};
	std::vector<std::pair<std::string, std::wstring>> DeviceList;
	std::vector<FormatInfo> CurDeviceFormats;
};

extern "C"
{


NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
{
	*outCount = (size_t)(1);
	if (!outFunctions)
		return NOS_RESULT_SUCCESS;

	NOS_RETURN_ON_FAILURE(RequestVulkanSubsystem());
		
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.webcam.WebcamIn"), WebcamInNode, outFunctions[0]);
	return NOS_RESULT_SUCCESS;
}

}
} // namespace nos::test
