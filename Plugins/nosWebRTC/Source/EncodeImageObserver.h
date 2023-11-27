#pragma once
#include <api/video_codecs/video_encoder.h>

class mzEncodeImageObserver : public webrtc::EncodedImageCallback {
public:
	mzEncodeImageObserver(std::function<void()> callback);
	// Inherited via EncodedImageCallback
	Result OnEncodedImage(const webrtc::EncodedImage& encoded_image, const webrtc::CodecSpecificInfo* codec_specific_info) override;
	std::function<void()> callback;
private:
};