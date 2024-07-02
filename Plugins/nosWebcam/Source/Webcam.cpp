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


		AddPinValueWatcher(NSN_Device, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				DevicePin = InterpretPinValue<char>(newVal);
				SelectedDeviceId = std::nullopt;
				CurDeviceFormats.clear();
				if (DevicePin != "NONE")
				{
					for (auto const& [name, id] : DeviceList)
						if (name == DevicePin)
						{
							SelectedDeviceId = id;
							break;
						}
					if (SelectedDeviceId)
						CurDeviceFormats = Cap.EnumerateFormats(*SelectedDeviceId);
					else
					{
						SetPinValue(NSN_Device, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
					}
				}
				if (!SelectedDeviceId)
					if (AutoSelectIfSingle(NSN_Device, GetDeviceList()))
						return;

				UpdateAfter(WebcamChangedPinType::Device, !oldValue);
			});

		AddPinValueWatcher(NSN_Format, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				FormatPin = InterpretPinValue<char>(newVal);
				SelectedFormatGuid = std::nullopt;
				if (FormatPin != "NONE")
				{
					SelectedFormatGuid = GetSubTypeFromFormatName(FormatPin);
					if (!SelectedFormatGuid)
					{
						SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(WebcamChangedPinType::FormatName, !oldValue);
			});

		AddPinValueWatcher(NSN_Resolution, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				ResolutionPin = InterpretPinValue<char>(newVal);
				SelectedResolution = std::nullopt;
				if (ResolutionPin != "NONE")
				{
					SelectedResolution = GetResolutionFromString(ResolutionPin);
					if (!SelectedResolution)
					{
						SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(WebcamChangedPinType::Resolution, !oldValue);
			});

		AddPinValueWatcher(NSN_FrameRate, [this](nos::Buffer const& newVal, std::optional<nos::Buffer> oldValue)
			{
				FrameRatePin = InterpretPinValue<char>(newVal);
				SelectedFrameRate = std::nullopt;
				if (FrameRatePin != "NONE")
				{
					SelectedFrameRate = GetFrameRateFromString(FrameRatePin);
					if (!SelectedFrameRate)
					{
						SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
						return;
					}
				}
				UpdateAfter(WebcamChangedPinType::FrameRate, !oldValue);
			});
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
			auto formatList = GetFormatList();
			UpdateStringList(GetFormatStringListName(), formatList);
			if (!SelectedDeviceId)
				SetPinValue(NSN_Format, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else
				AutoSelectIfSingle(NSN_Format, formatList);
			break;
		}
		case WebcamChangedPinType::FormatName:
		{
			UpdateStringList(GetResolutionStringListName(), GetResolutionList());
			if (!SelectedFormatGuid)
				SetPinValue(NSN_Resolution, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else
				AutoSelectIfSingle(NSN_Resolution, GetResolutionList());
			break;
		}
		case WebcamChangedPinType::Resolution:
		{
			UpdateStringList(GetFrameRateStringListName(), GetFrameRateList());
			if(!SelectedResolution)
				SetPinValue(NSN_FrameRate, nosBuffer{ .Data = (void*)"NONE", .Size = 5 });
			else
				AutoSelectIfSingle(NSN_FrameRate, GetFrameRateList());
			break;
		}
		case WebcamChangedPinType::FrameRate:
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

	bool AutoSelectIfSingle(nosName pinName, std::vector<std::string> const& list)
	{
		if (list.size() == 2)
		{
			SetPinValue(pinName, nosBuffer{ .Data = (void*)list[1].c_str(), .Size = list[1].size() + 1 });
			return true;
		}
		return false;
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
