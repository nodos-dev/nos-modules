// Copyright MediaZ AS. All Rights Reserved.

#include "VideoEncoderFactory.h"

nosVideoEncoderFactory::nosVideoEncoderFactory(nosEncodeImageObserver* observer) : observer(observer)
{
}

std::vector<webrtc::SdpVideoFormat> nosVideoEncoderFactory::GetSupportedFormats() const
{
    return { webrtc::SdpVideoFormat("VP8"), webrtc::SdpVideoFormat("VP9") };
}

webrtc::VideoEncoderFactory::CodecInfo nosVideoEncoderFactory::QueryVideoEncoder(const webrtc::SdpVideoFormat & format) const
{
    return CodecInfo();
}

std::unique_ptr<webrtc::VideoEncoder> nosVideoEncoderFactory::CreateVideoEncoder(const webrtc::SdpVideoFormat& format)
{
    return std::make_unique<nosVideoEncoder>(observer);
}
