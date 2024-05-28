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
struct DeviceInfo
{
	std::string Format;
	nos::fb::vec2u Resolution;
	nos::fb::vec2u DeltaSeconds;
};

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

class Capturer
{
public:
	Capturer();
	~Capturer();

	std::map<std::wstring, std::wstring> EnumerateDevices();
	DeviceInfo CreateDeviceFromName(std::wstring reqname);

	StreamSample ReadSample();

	ComPtr<struct IMFSourceReader> Reader{};
};

}