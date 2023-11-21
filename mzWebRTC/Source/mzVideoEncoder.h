#pragma once
#include <api/video_codecs/video_encoder.h>
#include "mzEncodeImageObserver.h"

class mzVideoEncoder : public webrtc::VideoEncoder {
public:
	mzVideoEncoder(mzEncodeImageObserver* callback);
	// Inherited via VideoEncoder
	virtual int InitEncode(const webrtc::VideoCodec* codec_settings, const VideoEncoder::Settings& settings);
	//virtual webrtc::VideoEncoder::EncoderInfo GetEncoderInfo() const override;
	int32_t RegisterEncodeCompleteCallback(webrtc::EncodedImageCallback* callback) override;
	int32_t Release() override;
	int32_t Encode(const webrtc::VideoFrame& frame, const std::vector<webrtc::VideoFrameType>* frame_types) override;
	void SetRates(const RateControlParameters& parameters) override;
private:
	std::unique_ptr<webrtc::VideoEncoder> internalEncoder;
	mzEncodeImageObserver* observer;
};