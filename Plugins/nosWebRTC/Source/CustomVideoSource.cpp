#include "mzCustomVideoSource.h"
#include <iostream>
#include "rtc_base/location.h"
mzCustomVideoSource::mzCustomVideoSource()
: CurrentState(webrtc::MediaSourceInterface::SourceState::kInitializing)
{

}

void mzCustomVideoSource::PushFrame(webrtc::VideoFrame& frame)
{
    CurrentState = webrtc::MediaSourceInterface::SourceState::kLive;
    // Broadcast the frame to all registered sinks
    OnFrame(frame);
}

void mzCustomVideoSource::AddRef() const {
    ref_count_.IncRef(); 
}

rtc::RefCountReleaseStatus mzCustomVideoSource::Release() const {
    const auto status = ref_count_.DecRef();
    if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
        delete this;
    }
    return status;
}
