/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once

#include <map>
#include <string>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wrl.h>

#include "Nodos/PluginHelpers.hpp"
namespace nos::webcam
{
template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

struct StreamSample
{
	ComPtr<struct IMFSample> Sample{};
	ComPtr<struct IMFMediaBuffer> Buffer{};
	uint8_t* Data = nullptr;
	DWORD Size = 0;

	StreamSample(ComPtr<struct IMFSample> sample);

	StreamSample(const StreamSample& other) = delete;
	StreamSample& operator=(const StreamSample& other) = delete;

	StreamSample(StreamSample&& other) noexcept;
	StreamSample& operator=(StreamSample&& other) noexcept;
	~StreamSample();
};

struct FormatInfo
{
	uint32_t StreamIndex;
	GUID MajorType;
	GUID SubType;
	nos::fb::vec2u Resolution;
	nos::fb::vec2u FrameRate;
	std::array<char, 4> GetFormatName() const
	{
		return { (char)(SubType.Data1 >> 0), (char)(SubType.Data1 >> 8), (char)(SubType.Data1 >> 16), (char)(SubType.Data1 >> 24) };
	}
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

class Capturer
{
public:
	Capturer();
	~Capturer();

	std::vector<std::pair<std::string, std::wstring>> EnumerateDevices();
	std::vector<FormatInfo> EnumerateFormats(std::wstring const& deviceId);
	std::optional<std::string> CreateStreamFromFormat(std::wstring const& deviceId, FormatInfo const& formatInfo);
	StreamSample ReadSample();
	void CloseStream();

	ComPtr<struct IMFSourceReader> Reader{};
};

}