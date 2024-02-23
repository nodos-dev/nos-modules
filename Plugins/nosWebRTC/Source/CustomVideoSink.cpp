// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "CustomVideoSink.h"

nosCustomVideoSink::nosCustomVideoSink()
{
}

void nosCustomVideoSink::SetOnFrameCallback(const std::function<void(const webrtc::VideoFrame&)>& callback)
{
	OnFrameCallback = callback;
}

void nosCustomVideoSink::OnFrame(const webrtc::VideoFrame& frame)
{
	if (OnFrameCallback) {
		OnFrameCallback(frame);
	}
}

void nosCustomVideoSink::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus nosCustomVideoSink::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}
