/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once

#include <unordered_map>
#include <string>
#include <expected>

#include <guiddef.h>
#include <wrl.h>

#include "Nodos/PluginHelpers.hpp"
#include "Webcam_generated.h"

struct IMFSourceReader;
struct IMFMediaType;
struct IMFSample;
struct IMFMediaBuffer;

namespace nos::webcam
{
template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

struct StreamSample
{
	ComPtr<IMFSample> Sample{};
	ComPtr<IMFMediaBuffer> Buffer{};
	uint8_t* Data = nullptr;
	DWORD Size = 0;

	StreamSample(ComPtr<IMFSample> sample);

	StreamSample(const StreamSample& other) = delete;
	StreamSample& operator=(const StreamSample& other) = delete;

	StreamSample(StreamSample&& other) noexcept;
	StreamSample& operator=(StreamSample&& other) noexcept;
	~StreamSample();
};

struct WebcamDevice
{
	std::string Name;
	std::wstring SymLink;
};

enum class WebcamFrameRate : uint32_t
{
	WEBCAM_FRAMERATE_1 = 0,
	WEBCAM_FRAMERATE_5,
	WEBCAM_FRAMERATE_7_5,
	WEBCAM_FRAMERATE_10,
	WEBCAM_FRAMERATE_14_98,
	WEBCAM_FRAMERATE_15,
	WEBCAM_FRAMERATE_20,
	WEBCAM_FRAMERATE_23_98,
	WEBCAM_FRAMERATE_24,
	WEBCAM_FRAMERATE_25,
	WEBCAM_FRAMERATE_29_97,
	WEBCAM_FRAMERATE_30,
	WEBCAM_FRAMERATE_47_95,
	WEBCAM_FRAMERATE_48,
	WEBCAM_FRAMERATE_50,
	WEBCAM_FRAMERATE_59_94,
	WEBCAM_FRAMERATE_60,
	WEBCAM_FRAMERATE_119_88,
	WEBCAM_FRAMERATE_120,
	COUNT
};

struct FormatInfo
{
	uint32_t StreamIndex;
	GUID MajorType;
	GUID SubType;
	nos::fb::vec2u Resolution;
	WebcamFrameRate FrameRate = WebcamFrameRate::WEBCAM_FRAMERATE_30;
	static FormatInfo FromMediaType(IMFMediaType* mediaType, uint32_t streamIndex);
	std::array<char, 4> GetFormatName() const
	{
		return { (char)(SubType.Data1 >> 0), (char)(SubType.Data1 >> 8), (char)(SubType.Data1 >> 16), (char)(SubType.Data1 >> 24) };
	}
};

struct WebcamStream
{
	WebcamStream(WebcamDevice const& device, ComPtr<IMFSourceReader> reader, uint32_t streamIndex);
	~WebcamStream();
	StreamSample ReadSample();
	void CloseStream();
	TWebcamStreamInfo GetStreamInfo() const;

	nosUUID StreamId;
	WebcamDevice Device;
	ComPtr<IMFSourceReader> Reader{};
	uint32_t StreamIndex = 0;
	ComPtr<IMFMediaType> MediaType{};
};

inline std::string GetFormatNameFromSubType(GUID const& subType)
{
	return std::string((char*)&subType.Data1, 4);
}
inline std::string GetResolutionString(nos::fb::vec2u const& resolution)
{
	return std::to_string(resolution.x()) + "x" + std::to_string(resolution.y());
}
inline const char* GetFrameRateString(WebcamFrameRate frameRate)
{
	switch (frameRate)
	{
		case WebcamFrameRate::WEBCAM_FRAMERATE_1:	    return "1";
		case WebcamFrameRate::WEBCAM_FRAMERATE_5:	    return "5";
		case WebcamFrameRate::WEBCAM_FRAMERATE_7_5:    return "7.5";
		case WebcamFrameRate::WEBCAM_FRAMERATE_10:	    return "10";
		case WebcamFrameRate::WEBCAM_FRAMERATE_14_98:  return "14.98";
		case WebcamFrameRate::WEBCAM_FRAMERATE_15:     return "15";
		case WebcamFrameRate::WEBCAM_FRAMERATE_20:     return "20";
		case WebcamFrameRate::WEBCAM_FRAMERATE_23_98:  return "23.98";
		case WebcamFrameRate::WEBCAM_FRAMERATE_24:     return "24";
		case WebcamFrameRate::WEBCAM_FRAMERATE_25:     return "25";
		case WebcamFrameRate::WEBCAM_FRAMERATE_29_97:  return "29.97";
		case WebcamFrameRate::WEBCAM_FRAMERATE_30:     return "30";
		case WebcamFrameRate::WEBCAM_FRAMERATE_47_95:  return "47.95";
		case WebcamFrameRate::WEBCAM_FRAMERATE_48:		return "48";
		case WebcamFrameRate::WEBCAM_FRAMERATE_50:     return "50";
		case WebcamFrameRate::WEBCAM_FRAMERATE_59_94:  return "59.94";
		case WebcamFrameRate::WEBCAM_FRAMERATE_60:     return "60";
		case WebcamFrameRate::WEBCAM_FRAMERATE_119_88: return "119.88";
		case WebcamFrameRate::WEBCAM_FRAMERATE_120:    return "120";
		default: DEBUG_BREAK;					        return "CUSTOM";
	}
}
std::optional<GUID> GetSubTypeFromFormatName(std::string const& formatName);
inline std::optional<nos::fb::vec2u> GetResolutionFromString(std::string const& resolution)
{
	auto pos = resolution.find('x');
	if (pos == std::string::npos)
		return std::nullopt;
	return nos::fb::vec2u{ std::stoul(resolution.substr(0, pos)), std::stoul(resolution.substr(pos + 1)) };

}

inline nos::fb::vec2u GetFrameRateVec2(WebcamFrameRate const& frameRate)
{
	switch (frameRate)
	{
		case WebcamFrameRate::WEBCAM_FRAMERATE_1:	    return { 1, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_5:	    return { 5, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_7_5:    return { 10000000, 1333333 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_10:	    return { 10, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_14_98:  return { 5000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_15:     return { 15, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_20:     return { 20, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_23_98:  return { 24000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_24:     return { 24, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_25:     return { 25, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_29_97:  return { 30000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_30:     return { 30, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_47_95:  return { 48000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_48:		return { 48, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_50:     return { 50, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_59_94:  return { 60000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_60:     return { 60, 1 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_119_88: return { 120000, 1001 };
		case WebcamFrameRate::WEBCAM_FRAMERATE_120:    return { 120, 1 };
		default: DEBUG_BREAK;					        return { 1, 50 }; break;
	}
}

inline std::optional<WebcamFrameRate> GetFrameRateFromString(std::string const& frameRate)
{
	for(uint32_t i = std::to_underlying(WebcamFrameRate::WEBCAM_FRAMERATE_1); i < std::to_underlying(WebcamFrameRate::COUNT); i++)
		if (frameRate == GetFrameRateString((WebcamFrameRate)i))
			return (WebcamFrameRate)i;
	return std::nullopt;
}

struct WebcamStreamManager
{
	static void Start();
	static void Stop();
	static WebcamStreamManager& GetInstance();

	static std::vector<WebcamDevice> EnumerateDevices();
	static std::vector<FormatInfo> EnumerateFormats(WebcamDevice const& device);
	std::expected<std::shared_ptr<WebcamStream>, std::string> OpenStreamFromFormat(WebcamDevice const& deviceId, FormatInfo const& formatInfo);
	void DeleteStream(nosUUID const& streamId);

	std::shared_ptr<WebcamStream> GetStream(nosUUID const& streamId);
private:
	static std::unique_ptr<WebcamStreamManager> Instance;
	std::shared_mutex OpenStreamsMutex;
	std::unordered_map<nosUUID, std::shared_ptr<WebcamStream>> OpenStreams;
};
}