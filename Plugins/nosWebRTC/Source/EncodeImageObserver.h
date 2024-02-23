/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once
#include <api/video_codecs/video_encoder.h>

class nosEncodeImageObserver : public webrtc::EncodedImageCallback {
public:
	nosEncodeImageObserver(std::function<void()> callback);
	// Inherited via EncodedImageCallback
	Result OnEncodedImage(const webrtc::EncodedImage& encoded_image, const webrtc::CodecSpecificInfo* codec_specific_info) override;
	std::function<void()> callback;
private:
};