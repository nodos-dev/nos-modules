
#define COBJMACROS 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <mfcaptureengine.h>

#include "MFCapture.h"

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
		CHECK(hr);
		nosEngine.WatchLog("ReadSample", sw.ElapsedString().c_str());
	}
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
		return StreamSample(nullptr);

	if (flags & MF_SOURCE_READERF_STREAMTICK)
		return StreamSample(nullptr);

	return StreamSample(pSample);
}

HRESULT EnumerateTypesForStream(IMFSourceReader* pReader, DWORD dwStreamIndex)
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
			// Get type's name etc.
			GUID guidMajorType;
			GUID guidSubType;
			UINT32 width = 0;
			UINT32 height = 0;
			UINT32 fps = 0;
			UINT32 bitrate = 0;
			UINT64 frameRate = 0;
			UINT32 frameRateDenom = 0;
			UINT32 frameRateNum = 0;
			UINT64 frameSize = 0;

			hr = pType->GetGUID(MF_MT_MAJOR_TYPE, &guidMajorType);
			hr = pType->GetGUID(MF_MT_SUBTYPE, &guidSubType);
			hr = pType->GetUINT64(MF_MT_FRAME_SIZE, &frameSize);
			hr = pType->GetUINT64(MF_MT_FRAME_RATE, &frameRate);

			width = frameSize >> 32;
			height = UINT32(frameSize & 0xFFFFFFFF);

			frameRateDenom = UINT32(frameRate);
			frameRateNum = UINT32(frameRate >> 32);

			std::string subTypeName = std::string(reinterpret_cast<char*>(&guidSubType.Data1), 4);

			nosEngine.LogI("Stream %d, Media Type %d: %s, Resolution: (%u, %u), FrameRate: %u/%u", dwStreamIndex, dwMediaTypeIndex, subTypeName.c_str(), width, height, frameRateNum, frameRateDenom);

			pType->Release();
		}
		++dwMediaTypeIndex;
	}
	return hr;
}

HRESULT EnumerateMediaTypes(IMFSourceReader* pReader)
{
	HRESULT hr = S_OK;
	DWORD dwStreamIndex = 0;

	while (SUCCEEDED(hr))
	{
		hr = EnumerateTypesForStream(pReader, dwStreamIndex);
		if (hr == MF_E_INVALIDSTREAMNUMBER)
		{
			hr = S_OK;
			break;
		}
		++dwStreamIndex;
	}
	return hr;
}

DeviceInfo Capturer::CreateDeviceFromName(std::wstring reqname)
{
	HRESULT hr;
	ComPtr<IMFMediaSource> pDevice = NULL;
	ComPtr<IMFAttributes> pAttrDevice = NULL;
	ComPtr<IMFAttributes> pAttrReader = NULL;
	DeviceInfo info{};

	hr = MFCreateAttributes(&pAttrDevice, 1);
	if (FAILED(hr)) return info;

	pAttrDevice->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	pAttrDevice->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, reqname.c_str());

	hr = MFCreateDeviceSource(pAttrDevice.Get(), &pDevice);
	if (FAILED(hr)) return info;

	ComPtr<IAMCameraControl> camControl;

	hr = pDevice->QueryInterface(IID_PPV_ARGS(&camControl));
	if (FAILED(hr)) return info;

	hr = camControl->Set(CameraControl_Exposure, -10, CameraControl_Flags_Manual);
	if (FAILED(hr)) return info;

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &Reader);
	if (FAILED(hr)) return info;

	EnumerateMediaTypes(Reader.Get());

	ComPtr<IMFMediaType> pTypeFormat;
	hr = MFCreateMediaType(&pTypeFormat);
	if (FAILED(hr)) return info;
	hr = pTypeFormat->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	hr = pTypeFormat->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_YUY2);
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_SIZE, ((UINT64)640 << 32) | 480);
	hr = pTypeFormat->SetUINT64(MF_MT_FRAME_RATE, ((UINT64)30 << 32) | 1);
	hr = pTypeFormat->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
	//// TODO:
	//IMFMediaType* pMediaType;
	//_reader->GetNativeMediaType(MF_SOURC
	// E_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READER_CURRENT_TYPE_INDEX, &pMediaType);
	// you can also set desired width/height here
	hr = Reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, pTypeFormat.Get());
	if (FAILED(hr)) return info;

	ComPtr<IMFMediaType> pType;
	hr = Reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pType);
	if (FAILED(hr)) return info;
	UINT64 tmp;
	hr = pType->GetUINT64(MF_MT_FRAME_SIZE, &tmp);
	info.Resolution = nos::fb::vec2u((UINT32)(tmp >> 32), (UINT32)(tmp));
	hr = pType->GetUINT64(MF_MT_FRAME_RATE, &tmp);
	info.DeltaSeconds = nos::fb::vec2u((UINT32)(tmp), (UINT32)(tmp >> 32));
	
	GUID guidSubType;
	hr = pType->GetGUID(MF_MT_SUBTYPE, &guidSubType);
	if(FAILED(hr)) return info;
	info.Format = std::string(reinterpret_cast<char*>(&guidSubType.Data1), 4);
	return info;
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
std::map<std::wstring, std::wstring> Capturer::EnumerateDevices()
{
	using namespace std;

	UINT32 count;
	RAIICoTask<IMFActivate**> devices = nullptr;
	map<wstring, wstring> result;
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

		bool cont = false;

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &length);

		if (FAILED(hr)) continue;

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &symlink, &length);
		if (FAILED(hr)) continue;
		result[*name] = symlink;

		devices[i]->Release();
		if (!cont)
			break;
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

}