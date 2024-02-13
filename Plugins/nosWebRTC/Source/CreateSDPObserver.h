/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once
#include "api/peer_connection_interface.h"
#include "api/scoped_refptr.h"

class nosCreateSDPObserver : public webrtc::CreateSessionDescriptionObserver {
public:
	nosCreateSDPObserver(int id);
	void SetSuccessCallback(std::function<void(webrtc::SessionDescriptionInterface*, int)> callback);
	void SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback);

	void AddRef() const override;
	rtc::RefCountReleaseStatus Release() const override;
private:
	void OnSuccess(webrtc::SessionDescriptionInterface* desc) override;
	void OnFailure(webrtc::RTCError error) override;

	std::function<void(webrtc::SessionDescriptionInterface*, int)> SuccessCallback;
	std::function<void(webrtc::RTCError, int)> FailureCallback;
	int peerConnectionID;
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };
};