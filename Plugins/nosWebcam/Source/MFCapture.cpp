
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

Capturor::Capturor()
{
	HRESULT  hr = CoInitialize(NULL);
	hr = MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET);
	CHECK(hr);

}
Capturor::~Capturor()
{
	MFShutdown();
	CoUninitialize();
}

StreamSample Capturor::ReadSample()
{
	HRESULT hr;
	DWORD streamIndex;
	DWORD flags;
	LONGLONG llTimeStamp;
	ComPtr<IMFSample> pSample = NULL;

	hr = Reader->ReadSample(
		MF_SOURCE_READER_FIRST_VIDEO_STREAM,
		0,
		&streamIndex,
		&flags,
		&llTimeStamp,
		&pSample
	);
	CHECK(hr);

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
			UINT32 frameRate = 0;
			UINT32 frameRateDenom = 0;
			UINT32 frameRateNum = 0;
			UINT32 frameSize = 0;

			hr = pType->GetGUID(MF_MT_MAJOR_TYPE, &guidMajorType);
			hr = pType->GetGUID(MF_MT_SUBTYPE, &guidSubType);
			hr = pType->GetUINT32(MF_MT_FRAME_SIZE, &frameSize);
			hr = pType->GetUINT32(MF_MT_FRAME_RATE, &frameRate);

			std::string subTypeName = std::string(reinterpret_cast<char*>(&guidSubType.Data1), 4);

			nosEngine.LogI("Stream %d, Media Type %d: %s", dwStreamIndex, dwMediaTypeIndex, subTypeName.c_str());

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

DeviceInfo Capturor::CreateDeviceFromName(std::wstring reqname)
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

	hr = MFCreateSourceReaderFromMediaSource(pDevice.Get(), NULL, &Reader);
	if (FAILED(hr)) return info;

	EnumerateMediaTypes(Reader.Get());

	ComPtr<IMFMediaType> pTypeFormat;
	hr = MFCreateMediaType(&pTypeFormat);
	if (FAILED(hr)) return info;
	hr = pTypeFormat->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
	hr = pTypeFormat->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_YUY2);
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


std::map<std::wstring, std::wstring> Capturor::EnumerateDevices()
{
	using namespace std;

	UINT32 count;
	IMFActivate** devices = 0;
	map<wstring, wstring> result;
	ComPtr<IMFAttributes> attr = 0;

	HRESULT hr;

	hr = MFCreateAttributes(&attr, 1);
	if (FAILED(hr)) goto cleanup;

	hr = attr->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	if (FAILED(hr)) goto cleanup;

	hr = MFEnumDeviceSources(attr.Get(), &devices, &count);
	if (FAILED(hr)) goto cleanup;


	for (UINT32 i = 0; i < count; i++)
	{
		UINT32 length;
		LPWSTR name;
		LPWSTR symlink;

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &length);
		if (FAILED(hr)) goto cleanup;

		hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &symlink, &length);
		if (FAILED(hr)) goto cleanup;

		result[name] = symlink;

		CoTaskMemFree(name);
		CoTaskMemFree(symlink);

		devices[i]->Release();
	}

cleanup:
	CoTaskMemFree(devices);
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