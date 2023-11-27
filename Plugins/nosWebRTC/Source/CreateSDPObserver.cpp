#include "CreateSDPObserver.h"

nosCreateSDPObserver::nosCreateSDPObserver(int id) : peerConnectionID(id)
{
}

void nosCreateSDPObserver::SetSuccessCallback(std::function<void(webrtc::SessionDescriptionInterface*, int)> callback)
{
	SuccessCallback = callback;
}

void nosCreateSDPObserver::SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback)
{
	FailureCallback = callback;
}

void nosCreateSDPObserver::OnSuccess(webrtc::SessionDescriptionInterface* desc)
{
	SuccessCallback(desc, peerConnectionID);
}

void nosCreateSDPObserver::OnFailure(webrtc::RTCError error)
{
	FailureCallback(error, peerConnectionID);
}

void nosCreateSDPObserver::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus nosCreateSDPObserver::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}
