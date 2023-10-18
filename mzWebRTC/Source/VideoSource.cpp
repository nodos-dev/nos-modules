#include "VideoSource.h"
#include <iostream>
#include "rtc_base/location.h"
CustomVideoSource::CustomVideoSource()
: CurrentState(webrtc::MediaSourceInterface::SourceState::kInitializing)
{

}

void CustomVideoSource::StartThread() {

}

void CustomVideoSource::PushFrame(webrtc::VideoFrame& frame)
{
    CurrentState = webrtc::MediaSourceInterface::SourceState::kLive;

    // Broadcast the frame to all registered sinks
    OnFrame(frame);

}

void CustomVideoSource::AddRef() const {
    ref_count_.IncRef(); 
}

rtc::RefCountReleaseStatus CustomVideoSource::Release() const {
    const auto status = ref_count_.DecRef();
    return status;
}
