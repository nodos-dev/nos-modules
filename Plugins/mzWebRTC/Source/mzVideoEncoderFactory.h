#pragma once
#include <api/video_codecs/builtin_video_encoder_factory.h>
#include "mzVideoEncoder.h"
#include "mzEncodeImageObserver.h"
class mzVideoEncoderFactory : public webrtc::VideoEncoderFactory {
public:
	// Inherited via VideoEncoderFactory
	mzVideoEncoderFactory(mzEncodeImageObserver* observer);
	std::vector<webrtc::SdpVideoFormat> GetSupportedFormats() const override;
	webrtc::VideoEncoderFactory::CodecInfo QueryVideoEncoder(const webrtc::SdpVideoFormat& format) const override;
	std::unique_ptr<webrtc::VideoEncoder> CreateVideoEncoder(const webrtc::SdpVideoFormat& format) override;
private:
	mzEncodeImageObserver* observer;
};