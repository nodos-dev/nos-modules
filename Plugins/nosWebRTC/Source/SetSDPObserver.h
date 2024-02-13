/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once
#include "api/peer_connection_interface.h"
#include "api/scoped_refptr.h"

class nosSetSDPObserver : public webrtc::SetSessionDescriptionObserver {
public:
	nosSetSDPObserver(int id);
	void SetSuccessCallback(std::function<void(int)> callback);
	void SetFailureCallback(std::function<void(webrtc::RTCError, int)> callback);

	void AddRef() const override;
	rtc::RefCountReleaseStatus Release() const override;
private:
	void OnSuccess() override;
	void OnFailure(webrtc::RTCError error) override;

	int peerConnectionID;
	std::function<void(int)> SuccessCallback;
	std::function<void(webrtc::RTCError,int)> FailureCallback;
	mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };
	
	// Inherited via SetSessionDescriptionObserver
};
