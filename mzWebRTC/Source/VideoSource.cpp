#include "VideoSource.h"
#include <iostream>
#include "rtc_base/location.h"
#include "libyuv/include/libyuv/convert.h"
CustomVideoSource::CustomVideoSource()
: CurrentState(webrtc::MediaSourceInterface::SourceState::kInitializing)
{

}
void CustomVideoSource::StartThread() {
}

void CustomVideoSource::MaybePushFrame()
{
          
}

void CustomVideoSource::PushFrame(uint8_t* data, int width, int height)
{
    //std::cout << "Frame pushed, frame count:  " << frameCount <<std::endl;
    CurrentState = webrtc::MediaSourceInterface::SourceState::kLive;
    
    rtc::scoped_refptr<webrtc::I420Buffer> buffer =
        webrtc::I420Buffer::Create(width, height);
    buffer->InitializeData();

    libyuv::ABGRToI420(data, 4 * width, buffer->MutableDataY(), buffer->StrideY(), buffer->MutableDataU(), buffer->StrideU(), buffer->MutableDataV(), buffer->StrideV(), width, height);

    webrtc::VideoFrame frame =
        webrtc::VideoFrame::Builder()
            .set_video_frame_buffer(buffer)
            .set_rotation(webrtc::kVideoRotation_0)
            .set_timestamp_us(rtc::TimeMicros())
            .build();

    // Broadcast the frame to all registered sinks
    OnFrame(frame);

}

void CustomVideoSource::AddRef() const {
    ref_count_.IncRef(); 
}

rtc::RefCountReleaseStatus CustomVideoSource::Release() const {
    const auto status = ref_count_.DecRef();
    if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
    delete this;
    }
    return status;
}
