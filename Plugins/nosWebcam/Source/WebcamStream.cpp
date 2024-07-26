// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "WebcamStream.h"

#define COBJMACROS 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <mfcaptureengine.h>
#include <locale>
#include <codecvt>

namespace nos::webcam
{
WebcamStream::WebcamStream(WebcamDevice const& device, ComPtr<IMFSourceReader> reader, uint32_t streamIndex) : Device(device), Reader(reader), StreamIndex(streamIndex)
{
	nosEngine.GenerateID(&StreamId);
	Reader->GetCurrentMediaType(streamIndex, &MediaType);
}

WebcamStream::~WebcamStream()
{
	CloseStream();
}

StreamSample WebcamStream::ReadSample()
{
	if (!Reader)
		return StreamSample(nullptr);
	HRESULT hr;
	DWORD streamIndex;
	DWORD flags;
	LONGLONG llTimeStamp;
	ComPtr<IMFSample> pSample = NULL;
	hr = Reader->ReadSample(
		StreamIndex,
		0,
		&streamIndex,
		&flags,
		&llTimeStamp,
		&pSample
	);
	if (FAILED(hr))
		return StreamSample(nullptr);
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
		return StreamSample(nullptr);
	if (flags & MF_SOURCE_READERF_STREAMTICK)
		return StreamSample(nullptr);
	return StreamSample(pSample);
}

void WebcamStream::CloseStream()
{
	if (Reader)
		Reader->Flush(StreamIndex);
	Reader.Reset();
}

TWebcamStreamInfo WebcamStream::GetStreamInfo() const
{
	TWebcamStreamInfo streamInfo{};
	streamInfo.id = std::make_unique<fb::UUID>(StreamId);
	streamInfo.device_name = Device.Name;
	FormatInfo formatInfo = FormatInfo::FromMediaType(MediaType.Get(), StreamIndex);
	streamInfo.format_name = std::string(formatInfo.GetFormatName().data(), 4);
	streamInfo.resolution = std::make_unique<fb::vec2u>(formatInfo.Resolution);
	streamInfo.frame_rate = std::make_unique<fb::vec2u>(GetFrameRateVec2(formatInfo.FrameRate));
	streamInfo.stream_index = StreamIndex;
	return streamInfo;
}

FormatInfo FormatInfo::FromMediaType(IMFMediaType* mediaType, uint32_t streamIndex)
{
	FormatInfo info{};
	info.StreamIndex = streamIndex;
	mediaType->GetGUID(MF_MT_MAJOR_TYPE, &info.MajorType);
	mediaType->GetGUID(MF_MT_SUBTYPE, &info.SubType);
	UINT64 frameSize = 0;
	mediaType->GetUINT64(MF_MT_FRAME_SIZE, &frameSize);
	UINT64 frameRate = 0;
	mediaType->GetUINT64(MF_MT_FRAME_RATE, &frameRate);

	info.Resolution = nos::fb::vec2u(frameSize >> 32, frameSize & 0xFFFFFFFF);
	auto frameRateVec2 = nos::fb::vec2u(frameRate >> 32, frameRate & 0xFFFFFFFF);
	for(uint32_t i = std::to_underlying(WebcamFrameRate::WEBCAM_FRAMERATE_1); i < std::to_underlying(WebcamFrameRate::COUNT); i++)
		if (frameRateVec2 == GetFrameRateVec2(WebcamFrameRate(i)))
		{
			info.FrameRate = WebcamFrameRate(i);
			break;
		}

	return info;
}

StreamSample::StreamSample(ComPtr<IMFSample> sample) : Sample(sample)
{
	if (sample != 0)
	{
		sample->GetBufferByIndex(0, &Buffer);
		Buffer->Lock((BYTE**)&Data, NULL, NULL);
		Buffer->GetCurrentLength(&Size);
	}
}

StreamSample::StreamSample(StreamSample&& other) noexcept = default;
StreamSample& StreamSample::operator=(StreamSample&& other) noexcept = default;

StreamSample::~StreamSample()
{
	if (Buffer != 0)
	{
		Buffer->Unlock();
	}
}

std::optional<GUID> GetSubTypeFromFormatName(std::string const& formatName)
{
	if (formatName.size() != 4 || formatName == "NONE")
	{
		return std::nullopt;
	}
	GUID base = MFVideoFormat_Base;
	base.Data1 = *reinterpret_cast<const DWORD*>(formatName.c_str());
	return base;
}

template<typename T>
struct RAIICoTask
{
	T Ptr;
	RAIICoTask(T ptr = {}) : Ptr(ptr) {}
	~RAIICoTask()
	{
		CoTaskMemFree(*this);
	}

	T& operator*() { return Ptr; }
	T operator->() { return Ptr; }
	operator T() { return Ptr; }
	T* operator&() { return &Ptr; }

};
void WebcamStreamManager::Start()
{
	HRESULT hr = CoInitialize(NULL);
	hr = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET);
	Instance = std::make_unique<WebcamStreamManager>();
}
void WebcamStreamManager::Stop()
{
	if (Instance)
	{
		std::unique_lock lock(Instance->OpenStreamsMutex);
		for (auto& stream : Instance->OpenStreams)
		{
			stream.second->CloseStream();
		}
		Instance->OpenStreams.clear();
		Instance.reset();
	}
	MFShutdown();
	CoUninitialize();
}

std::unique_ptr<WebcamStreamManager> WebcamStreamManager::Instance = nullptr;
WebcamStreamManager& WebcamStreamManager::GetInstance()
{
	return *Instance;
}
std::vector<WebcamDevice> WebcamStreamManager::EnumerateDevices()
{
	using namespace std;

	UINT32 count;
	RAIICoTask<IMFActivate**> devices = nullptr;
	std::vector<WebcamDevice> result;
	ComPtr<IMFAttributes> attr = 0;

	HRESULT hr;

	hr = MFCreateAttributes(&attr, 1);
	if (FAILED(hr)) return result;

	hr = attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	if (FAILED(hr)) return result;

	hr = MFEnumDeviceSources(attr.Get(), &devices, &count);
	if (FAILED(hr)) return result;


	for (UINT32 i = 0; i < count; i++)
	{
		UINT32 length;
		RAIICoTask<LPWSTR> name;
		RAIICoTask<LPWSTR> symlink;

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &length);

		if (FAILED(hr)) continue;

		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::string nameNarrow = converter.to_bytes(*name);

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &symlink, &length);
		if (FAILED(hr)) continue;
		result.emplace_back(nameNarrow, std::wstring(symlink));

		devices[i]->Release();
	}

	return result;
}


HRESULT EnumerateTypesForStream(IMFSourceReader* pReader, DWORD dwStreamIndex, std::vector<FormatInfo>& types)
{
	HRESULT hr = S_OK;
	DWORD dwMediaTypeIndex = 0;

	while (SUCCEEDED(hr))
	{
		IMFMediaType* pType = NULL;
		hr = pReader->GetNativeMediaType(dwStreamIndex, dwMediaTypeIndex, &pType);
		if (hr == MF_E_NO_MORE_TYPES)
		{
			hr = S_OK;
			break;
		}
		else if (SUCCEEDED(hr))
		{
			FormatInfo mediaInfo = FormatInfo::FromMediaType(pType, dwStreamIndex);
			if (mediaInfo.SubType == MFVideoFormat_YUY2 || mediaInfo.SubType == MFVideoFormat_NV12)
				types.push_back(mediaInfo);
			pType->Release();
		}
		++dwMediaTypeIndex;
	}
	return hr;
}

std::vector<FormatInfo> EnumerateMediaTypes(IMFSourceReader* pReader)
{
	std::vector<FormatInfo> types;
	HRESULT hr = S_OK;
	DWORD dwStreamIndex = 0;

	while (SUCCEEDED(hr))
	{
		hr = EnumerateTypesForStream(pReader, dwStreamIndex, types);
		if (hr == MF_E_INVALIDSTREAMNUMBER)
		{
			break;
		}
		++dwStreamIndex;
	}

	// Order formats based on NV12 > YUY2, then resolution, then frame rate
	std::sort(types.begin(), types.end(), [](FormatInfo const& a, FormatInfo const& b) {
		if (a.SubType != b.SubType)
			return a.SubType == MFVideoFormat_NV12;
		if (a.Resolution.x() != b.Resolution.x())
			return a.Resolution.x() > b.Resolution.x();
		if (a.Resolution.y() != b.Resolution.y())
			return a.Resolution.y() > b.Resolution.y();
		if (std::to_underlying(a.FrameRate) != std::to_underlying(b.FrameRate))
			return std::to_underlying(a.FrameRate) > std::to_underlying(b.FrameRate);
		return false;
		});

	return types;
}

std::vector<FormatInfo> WebcamStreamManager::EnumerateFormats(WebcamDevice const& device)
{
	HRESULT hr;
	ComPtr<IMFMediaSource> pDevice = NULL;
	ComPtr<IMFAttributes> pAttrDevice = NULL;
	ComPtr<IMFAttributes> pAttrReader = NULL;
	std::vector<FormatInfo> formats;

	hr = MFCreateAttributes(&pAttrDevice, 1);
	if (FAILED(hr)) return formats;

	pAttrDevice->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	pAttrDevice->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, device.SymLink.c_str());

	hr = MFCreateDeviceSource(pAttrDevice.Get(), &pDevice);
	if (FAILED(hr)) return formats;

	ComPtr<IMFSourceReader> reader = NULL;

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &reader);
	if (FAILED(hr)) return formats;

	formats = EnumerateMediaTypes(reader.Get());

	return formats;
}

std::string GetLastErrorAsString(HRESULT err)
{
	if (err == 0) {
		return std::string();
	}
	LPSTR messageBuffer = nullptr;
	size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
	std::string message(messageBuffer, size);
	LocalFree(messageBuffer);
	return message;
}

std::expected<std::shared_ptr<WebcamStream>, std::string> WebcamStreamManager::OpenStreamFromFormat(WebcamDevice const& device, FormatInfo const& formatInfo)
{
	std::expected<std::shared_ptr<WebcamStream>, std::string> result;
	HRESULT hr;
	ComPtr<IMFMediaSource> pDevice = NULL;
	ComPtr<IMFAttributes> pAttrDevice = NULL;
	ComPtr<IMFAttributes> pAttrReader = NULL;
	hr = MFCreateAttributes(&pAttrDevice, 1);
	if (FAILED(hr)) return std::unexpected(GetLastErrorAsString(hr));

	pAttrDevice->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	pAttrDevice->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, device.SymLink.c_str());

	hr = MFCreateDeviceSource(pAttrDevice.Get(), &pDevice);
	if (FAILED(hr)) return std::unexpected(GetLastErrorAsString(hr));

	ComPtr<IMFSourceReader> reader = NULL;

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &reader);
	if (FAILED(hr)) return std::unexpected(GetLastErrorAsString(hr));

	ComPtr<IMFMediaType> pTypeFormat;
	hr = MFCreateMediaType(&pTypeFormat);
	if (FAILED(hr)) return std::unexpected(GetLastErrorAsString(hr));
	hr = pTypeFormat->SetGUID(MF_MT_MAJOR_TYPE, formatInfo.MajorType);
	hr = pTypeFormat->SetGUID(MF_MT_SUBTYPE, formatInfo.SubType);
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_SIZE, ((UINT64)formatInfo.Resolution.x() << 32) | formatInfo.Resolution.y());
	nos::fb::vec2u frameRate = GetFrameRateVec2(formatInfo.FrameRate);
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_RATE, ((UINT64)frameRate.x() << 32) | frameRate.y());
	hr = pTypeFormat->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);

	hr = reader->SetCurrentMediaType(formatInfo.StreamIndex, NULL, pTypeFormat.Get());
	if (FAILED(hr)) return std::unexpected(GetLastErrorAsString(hr));

	ComPtr<IMFMediaType> pGetMediaType;
	hr = reader->GetCurrentMediaType(formatInfo.StreamIndex, &pGetMediaType);
	FormatInfo info = FormatInfo::FromMediaType(pGetMediaType.Get(), formatInfo.StreamIndex);

	std::shared_ptr<WebcamStream> stream = std::make_shared<WebcamStream>(device, reader, formatInfo.StreamIndex);
	std::unique_lock lock(OpenStreamsMutex);
	OpenStreams[stream->StreamId] = stream;
	return stream;
}

void WebcamStreamManager::DeleteStream(nosUUID const& streamId)
{
	std::unique_lock lock(OpenStreamsMutex);
	if (auto it = OpenStreams.find(streamId); it != OpenStreams.end())
	{
		OpenStreams.erase(it);
	}
}

std::shared_ptr<WebcamStream> WebcamStreamManager::GetStream(nosUUID const& streamId)
{
	std::shared_lock lock(OpenStreamsMutex);
	if (auto it = OpenStreams.find(streamId); it != OpenStreams.end())
		return it->second;
	return nullptr;
}

}; // namespace nos::webcam