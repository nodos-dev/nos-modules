#include "mzSetSDPObserver.h"

mzSetSDPObserver::mzSetSDPObserver(int id) : peerConnectionID(id)
{
}

void mzSetSDPObserver::SetSuccessCallback(std::function<void(int)> callback)
{
	SuccessCallback = callback;
}

void mzSetSDPObserver::SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback)
{
	FailureCallback = callback;
}

void mzSetSDPObserver::OnSuccess()
{
	SuccessCallback(peerConnectionID);
}

void mzSetSDPObserver::OnFailure(webrtc::RTCError error)
{
	FailureCallback(error, peerConnectionID);
}

void mzSetSDPObserver::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus mzSetSDPObserver::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}