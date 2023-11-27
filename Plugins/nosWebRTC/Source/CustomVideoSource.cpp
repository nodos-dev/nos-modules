#include "CustomVideoSource.h"
#include <iostream>
#include "rtc_base/location.h"
nosCustomVideoSource::nosCustomVideoSource()
: CurrentState(webrtc::MediaSourceInterface::SourceState::kInitializing)
{

}

void nosCustomVideoSource::PushFrame(webrtc::VideoFrame& frame)
{
    CurrentState = webrtc::MediaSourceInterface::SourceState::kLive;
    // Broadcast the frame to all registered sinks
    OnFrame(frame);
}

void nosCustomVideoSource::AddRef() const {
    ref_count_.IncRef(); 
}

rtc::RefCountReleaseStatus nosCustomVideoSource::Release() const {
    const auto status = ref_count_.DecRef();
    if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
        delete this;
    }
    return status;
}
