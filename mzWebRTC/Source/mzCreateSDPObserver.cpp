#include "mzCreateSDPObserver.h"

mzCreateSDPObserver::mzCreateSDPObserver(int id) : peerConnectionID(id)
{
}

void mzCreateSDPObserver::SetSuccessCallback(std::function<void(webrtc::SessionDescriptionInterface*, int)> callback)
{
	SuccessCallback = callback;
}

void mzCreateSDPObserver::SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback)
{
	FailureCallback = callback;
}

void mzCreateSDPObserver::OnSuccess(webrtc::SessionDescriptionInterface* desc)
{
	SuccessCallback(desc, peerConnectionID);
}

void mzCreateSDPObserver::OnFailure(webrtc::RTCError error)
{
	FailureCallback(error, peerConnectionID);
}

void mzCreateSDPObserver::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus mzCreateSDPObserver::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}
