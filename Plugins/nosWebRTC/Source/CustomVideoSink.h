#pragma once
#include <media/base/video_common.h>
#include <api/video/video_frame.h>
#include <api/media_stream_interface.h>
#include "mzI420Buffer.h"
class mzCustomVideoSink : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
public:
	mzCustomVideoSink();
	void SetOnFrameCallback(const std::function<void(const webrtc::VideoFrame&)>& callback);
	// Inherited via VideoSinkInterface
	void OnFrame(const webrtc::VideoFrame& frame) override;
	bool IsAvailable = true;
	void AddRef() const;
	rtc::RefCountReleaseStatus Release() const;
private:
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };
	std::function<void(const webrtc::VideoFrame&)> OnFrameCallback;

};