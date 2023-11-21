#include "mzVideoEncoderFactory.h"

mzVideoEncoderFactory::mzVideoEncoderFactory(mzEncodeImageObserver* observer) : observer(observer)
{
}

std::vector<webrtc::SdpVideoFormat> mzVideoEncoderFactory::GetSupportedFormats() const
{
    return { webrtc::SdpVideoFormat("VP8"), webrtc::SdpVideoFormat("VP9") };
}

webrtc::VideoEncoderFactory::CodecInfo mzVideoEncoderFactory::QueryVideoEncoder(const webrtc::SdpVideoFormat & format) const
{
    return CodecInfo();
}

std::unique_ptr<webrtc::VideoEncoder> mzVideoEncoderFactory::CreateVideoEncoder(const webrtc::SdpVideoFormat& format)
{
    return std::make_unique<mzVideoEncoder>(observer);
}
