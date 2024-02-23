// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "SetSDPObserver.h"

nosSetSDPObserver::nosSetSDPObserver(int id) : peerConnectionID(id)
{
}

void nosSetSDPObserver::SetSuccessCallback(std::function<void(int)> callback)
{
	SuccessCallback = callback;
}

void nosSetSDPObserver::SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback)
{
	FailureCallback = callback;
}

void nosSetSDPObserver::OnSuccess()
{
	SuccessCallback(peerConnectionID);
}

void nosSetSDPObserver::OnFailure(webrtc::RTCError error)
{
	FailureCallback(error, peerConnectionID);
}

void nosSetSDPObserver::AddRef() const
{
	ref_count_.IncRef();
}

rtc::RefCountReleaseStatus nosSetSDPObserver::Release() const
{
	const auto status = ref_count_.DecRef();
	if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
		delete this;
	}
	return status;
}