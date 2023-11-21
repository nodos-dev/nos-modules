#include "mzCustomVideoSink.h"

mzCustomVideoSink::mzCustomVideoSink()
{
}

void mzCustomVideoSink::SetOnFrameCallback(const std::function<void(const webrtc::VideoFrame&)>& callback)
{
	OnFrameCallback = callback;
}

void mzCustomVideoSink::OnFrame(const webrtc::VideoFrame& frame)
{
	if (OnFrameCallback) {
		OnFrameCallback(frame);
	}
}

void mzCustomVideoSink::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus mzCustomVideoSink::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}
