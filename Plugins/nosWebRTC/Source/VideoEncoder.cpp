// Copyright MediaZ AS. All Rights Reserved.

#include "VideoEncoder.h"
#include <api/video_codecs/builtin_video_encoder_factory.h>


nosVideoEncoder::nosVideoEncoder(nosEncodeImageObserver* callback)
{
    auto videoFactory = webrtc::CreateBuiltinVideoEncoderFactory();
    internalEncoder = videoFactory->CreateVideoEncoder(webrtc::SdpVideoFormat("VP8"));
    observer = callback;
}

int nosVideoEncoder::InitEncode(const webrtc::VideoCodec* codec_settings, const VideoEncoder::Settings& settings)
{
    internalEncoder->InitEncode(codec_settings, settings);
    return 0;
}

int32_t nosVideoEncoder::RegisterEncodeCompleteCallback(webrtc::EncodedImageCallback* callback)
{
    internalEncoder->RegisterEncodeCompleteCallback(callback);
    return 0;
}

int32_t nosVideoEncoder::Release()
{
    return 0;
}

int32_t nosVideoEncoder::Encode(const webrtc::VideoFrame& frame, const std::vector<webrtc::VideoFrameType>* frame_types)
{
    internalEncoder->Encode(frame, frame_types);
    observer->callback();
    return 0;
}

void nosVideoEncoder::SetRates(const RateControlParameters& parameters)
{
    //TOOD: We may specify bandowth, max fps, etc in here!
}
