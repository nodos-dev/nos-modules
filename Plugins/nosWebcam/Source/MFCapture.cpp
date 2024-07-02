
#define COBJMACROS 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <mfcaptureengine.h>
#include "MFCapture.h"
#include <locale>
#include <codecvt>

#include <map>
#include <string>
#include <ranges>
#include <exception>
#include <shlwapi.h>

#include <strmif.h>

#include <nosUtil/Stopwatch.hpp>


#define CHECK(hr) if(FAILED(hr)) throw std::exception("fail");
#define CHECK(hr) if(FAILED(hr)) throw std::exception("fail");

#include <Nodos/PluginHelpers.hpp>

namespace nos::webcam
{

template <class T> void SafeRelease(T** ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = NULL;
	}
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

Capturer::Capturer()
{
	HRESULT  hr = CoInitialize(NULL);
	hr = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET);
	CHECK(hr);

}
Capturer::~Capturer()
{
	MFShutdown();
	CoUninitialize();
}

StreamSample Capturer::ReadSample()
{
	if(!Reader)
		return StreamSample(nullptr);
	HRESULT hr;
	DWORD streamIndex;
	DWORD flags;
	LONGLONG llTimeStamp;
	ComPtr<IMFSample> pSample = NULL;

	{

		util::Stopwatch sw;
		hr = Reader->ReadSample(
			MF_SOURCE_READER_FIRST_VIDEO_STREAM,
			0,
			&streamIndex,
			&flags,
			&llTimeStamp,
			&pSample
		);
		if (FAILED(hr))
			return StreamSample(nullptr);
		nosEngine.WatchLog("ReadSample", sw.ElapsedString().c_str());
	}
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
		return StreamSample(nullptr);

	if (flags & MF_SOURCE_READERF_STREAMTICK)
		return StreamSample(nullptr);

	return StreamSample(pSample);
}

void Capturer::CloseStream()
{
	if (Reader)
		Reader->Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
	Reader.Reset();
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
			FormatInfo mediaInfo = {};
			mediaInfo.StreamIndex = dwStreamIndex;
			hr = pType->GetGUID(MF_MT_MAJOR_TYPE, &mediaInfo.MajorType);
			hr = pType->GetGUID(MF_MT_SUBTYPE, &mediaInfo.SubType);
			UINT64 frameSize = 0;
			hr = pType->GetUINT64(MF_MT_FRAME_SIZE, &frameSize);
			UINT64 frameRate = 0;
			hr = pType->GetUINT64(MF_MT_FRAME_RATE, &frameRate);

			mediaInfo.Resolution = nos::fb::vec2u(frameSize >> 32, frameSize & 0xFFFFFFFF);
			mediaInfo.FrameRate = nos::fb::vec2u(UINT32(frameRate), UINT32(frameRate >> 32));

			if(mediaInfo.SubType == MFVideoFormat_YUY2)
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
	return types;
}

std::vector<FormatInfo> Capturer::EnumerateFormats(std::wstring const& deviceId)
{
	HRESULT hr;
	ComPtr<IMFMediaSource> pDevice = NULL;
	ComPtr<IMFAttributes> pAttrDevice = NULL;
	ComPtr<IMFAttributes> pAttrReader = NULL;
	std::vector<FormatInfo> formats;

	hr = MFCreateAttributes(&pAttrDevice, 1);
	if (FAILED(hr)) return formats;

	pAttrDevice->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	pAttrDevice->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, deviceId.c_str());

	hr = MFCreateDeviceSource(pAttrDevice.Get(), &pDevice);
	if (FAILED(hr)) return formats;

	ComPtr<IMFSourceReader> reader = NULL;

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &reader);
	if (FAILED(hr)) return formats;

	formats = EnumerateMediaTypes(reader.Get());

	return formats;
}

std::optional<std::string> Capturer::CreateStreamFromFormat(std::wstring const& deviceId, FormatInfo const& formatInfo)
{
	HRESULT hr;
	ComPtr<IMFMediaSource> pDevice = NULL;
	ComPtr<IMFAttributes> pAttrDevice = NULL;
	ComPtr<IMFAttributes> pAttrReader = NULL;

	hr = MFCreateAttributes(&pAttrDevice, 1);
	if (FAILED(hr)) return GetLastErrorAsString(hr);

	pAttrDevice->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	pAttrDevice->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, deviceId.c_str());

	hr = MFCreateDeviceSource(pAttrDevice.Get(), &pDevice);
	if (FAILED(hr)) return GetLastErrorAsString(hr);

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &Reader);
	if (FAILED(hr)) return GetLastErrorAsString(hr);

	ComPtr<IMFMediaType> pTypeFormat;
	hr = MFCreateMediaType(&pTypeFormat);
	if (FAILED(hr)) return GetLastErrorAsString(hr);
	hr = pTypeFormat->SetGUID(MF_MT_MAJOR_TYPE, formatInfo.MajorType);
	hr = pTypeFormat->SetGUID(MF_MT_SUBTYPE, formatInfo.SubType);
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_SIZE, ((UINT64)formatInfo.Resolution.x() << 32) | formatInfo.Resolution.y());
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_RATE, ((UINT64)formatInfo.FrameRate.x() << 32) | formatInfo.FrameRate.y());
	hr = pTypeFormat->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
	


	hr = Reader->SetCurrentMediaType(formatInfo.StreamIndex, NULL, pTypeFormat.Get());\
	if (FAILED(hr)) return GetLastErrorAsString(hr);

	return std::nullopt;
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
std::vector<std::pair<std::string, std::wstring>> Capturer::EnumerateDevices()
{
	using namespace std;

	UINT32 count;
	RAIICoTask<IMFActivate**> devices = nullptr;
	std::vector<std::pair<string, wstring>> result;
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
		result.emplace_back(nameNarrow, symlink);

		devices[i]->Release();
	}

	return result;
}

StreamSample::StreamSample(ComPtr<struct IMFSample> sample) : Sample(sample)
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

}
