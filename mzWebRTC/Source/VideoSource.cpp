#include "VideoSource.h"
#include <iostream>
#include "rtc_base/location.h"

CustomVideoSource::CustomVideoSource()
: CurrentState(webrtc::MediaSourceInterface::SourceState::kInitializing)
{
    StartThread();
}
void CustomVideoSource::StartThread() {
    thread = rtc::Thread::Create().release();
    thread->Start();
    thread->PostTask(RTC_FROM_HERE,[this]() { MaybePushFrame(); });
}

void CustomVideoSource::MaybePushFrame()
{
    while (!thread->IsQuitting()) {
    PushFrame();
    thread->SleepMs(50);
    }
          

}

void CustomVideoSource::PushFrame()
{
    static int frameCount = 1;

    if (frameCount >= 255)
        frameCount = 1;
    //std::cout << "Frame pushed, frame count:  " << frameCount <<std::endl;
    CurrentState = webrtc::MediaSourceInterface::SourceState::kLive;
    int width = 640;
    int height = 480;

    rtc::scoped_refptr<webrtc::I420Buffer> buffer =
        webrtc::I420Buffer::Create(width, height);
    buffer->InitializeData();

    memset(buffer->MutableDataY(), frameCount,
            buffer->StrideY() * height);
    memset(buffer->MutableDataU(), frameCount,
            buffer->StrideU() *
                ((height + 1) / 2)); 
    memset(buffer->MutableDataV(), 255 - frameCount,
            buffer->StrideV() *
                ((height + 1) / 2));
    frameCount++;
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
