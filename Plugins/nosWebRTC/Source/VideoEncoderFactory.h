/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once
#include <api/video_codecs/builtin_video_encoder_factory.h>
#include "VideoEncoder.h"
#include "EncodeImageObserver.h"
class nosVideoEncoderFactory : public webrtc::VideoEncoderFactory {
public:
	// Inherited via VideoEncoderFactory
	nosVideoEncoderFactory(nosEncodeImageObserver* observer);
	std::vector<webrtc::SdpVideoFormat> GetSupportedFormats() const override;
	webrtc::VideoEncoderFactory::CodecInfo QueryVideoEncoder(const webrtc::SdpVideoFormat& format) const override;
	std::unique_ptr<webrtc::VideoEncoder> CreateVideoEncoder(const webrtc::SdpVideoFormat& format) override;
private:
	nosEncodeImageObserver* observer;
};