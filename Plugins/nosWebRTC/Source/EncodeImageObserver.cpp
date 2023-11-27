#include "EncodeImageObserver.h"

nosEncodeImageObserver::nosEncodeImageObserver(std::function<void()> callback) : callback(callback)
{
}

webrtc::EncodedImageCallback::Result nosEncodeImageObserver::OnEncodedImage(const webrtc::EncodedImage& encoded_image, const webrtc::CodecSpecificInfo* codec_specific_info)
{
    callback();
    return Result({webrtc::EncodedImageCallback::Result::Error::OK});
}
