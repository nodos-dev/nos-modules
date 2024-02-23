/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once
#include <media/base/video_common.h>
#include <api/video/video_frame.h>
#include <api/media_stream_interface.h>
#include "I420Buffer.h"
class nosCustomVideoSink : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
public:
	nosCustomVideoSink();
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