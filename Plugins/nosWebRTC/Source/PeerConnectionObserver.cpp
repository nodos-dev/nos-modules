// Copyright MediaZ AS. All Rights Reserved.

#include "PeerConnectionObserver.h"
nosPeerConnectionObserver::nosPeerConnectionObserver(int _peerConnectionID) :peerConnectionID(_peerConnectionID)
{
}

void nosPeerConnectionObserver::OnSignalingChange(webrtc::PeerConnectionInterface::SignalingState new_state)
{
	SignalingChangeCallback(new_state, peerConnectionID);
}

void nosPeerConnectionObserver::OnAddTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver, const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>& streams)
{
	AddTrackCallback(receiver, streams, peerConnectionID);
}

void nosPeerConnectionObserver::OnRemoveTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver)
{
	RemoveTrackCallback(receiver, peerConnectionID);
}

void nosPeerConnectionObserver::OnDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> data_channel)
{
	DataChannelCallback(data_channel, peerConnectionID);
}

void nosPeerConnectionObserver::OnRenegotiationNeeded()
{
	RenegotiationNeededCallback(peerConnectionID);
}

void nosPeerConnectionObserver::OnIceConnectionChange(webrtc::PeerConnectionInterface::IceConnectionState new_state)
{
	IceConnectionChangeCallback(new_state, peerConnectionID);
}

void nosPeerConnectionObserver::OnIceGatheringChange(webrtc::PeerConnectionInterface::IceGatheringState new_state)
{
	IceGatheringChangeCallback(new_state, peerConnectionID);
}

void nosPeerConnectionObserver::OnIceCandidate(const webrtc::IceCandidateInterface* candidate)
{
	IceCandidateCallback(candidate, peerConnectionID);
}

void nosPeerConnectionObserver::SetSignalingChangeCallback(std::function<void(webrtc::PeerConnectionInterface::SignalingState, int)> callback)
{
	SignalingChangeCallback = callback;
}

void nosPeerConnectionObserver::SetAddTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>&, int)> callback)
{
	AddTrackCallback = callback;
}

void nosPeerConnectionObserver::SetRemoveTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, int)> callback)
{
	RemoveTrackCallback = callback;
}

void nosPeerConnectionObserver::SetDataChannelCallback(std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>, int)> callback)
{
	DataChannelCallback = callback;
}

void nosPeerConnectionObserver::SetRenegotiationCallback(std::function<void(int)> callback)
{
	RenegotiationNeededCallback = callback;
}

void nosPeerConnectionObserver::SetICEConnectionChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceConnectionState, int)> callback)
{
	IceConnectionChangeCallback = callback;
}

void nosPeerConnectionObserver::SetICEGatheringChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceGatheringState, int)> callback)
{
	IceGatheringChangeCallback = callback;
}

void nosPeerConnectionObserver::SetICECandidateCallback(std::function<void(const webrtc::IceCandidateInterface*, int)> callback)
{
	IceCandidateCallback = callback;
}
