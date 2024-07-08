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

struct FormatInfo
{
	uint32_t StreamIndex;
	GUID MajorType;
	GUID SubType;
	nos::fb::vec2u Resolution;
	nos::fb::vec2u FrameRate;
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
inline std::string GetFrameRateString(nos::fb::vec2u const& frameRate)
{
	return std::to_string(frameRate.x()) + "/" + std::to_string(frameRate.y());
}
std::optional<GUID> GetSubTypeFromFormatName(std::string const& formatName);
inline std::optional<nos::fb::vec2u> GetResolutionFromString(std::string const& resolution)
{
	auto pos = resolution.find('x');
	if (pos == std::string::npos)
		return std::nullopt;
	return nos::fb::vec2u{ std::stoul(resolution.substr(0, pos)), std::stoul(resolution.substr(pos + 1)) };

}
inline std::optional<nos::fb::vec2u> GetFrameRateFromString(std::string const& frameRate)
{
	auto pos = frameRate.find('/');
	if (pos == std::string::npos)
		return std::nullopt;;
	return nos::fb::vec2u{ std::stoul(frameRate.substr(0, pos)), std::stoul(frameRate.substr(pos + 1)) };
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