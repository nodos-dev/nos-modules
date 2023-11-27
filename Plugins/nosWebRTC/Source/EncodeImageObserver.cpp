#include "mzEncodeImageObserver.h"

mzEncodeImageObserver::mzEncodeImageObserver(std::function<void()> callback) : callback(callback)
{
}

webrtc::EncodedImageCallback::Result mzEncodeImageObserver::OnEncodedImage(const webrtc::EncodedImage& encoded_image, const webrtc::CodecSpecificInfo* codec_specific_info)
{
    callback();
    return Result({webrtc::EncodedImageCallback::Result::Error::OK});
}
