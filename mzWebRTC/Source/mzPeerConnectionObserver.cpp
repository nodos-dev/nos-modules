#include "mzPeerConnectionObserver.h"
mzPeerConnectionObserver::mzPeerConnectionObserver(int _peerConnectionID) :peerConnectionID(_peerConnectionID)
{
}

void mzPeerConnectionObserver::OnSignalingChange(webrtc::PeerConnectionInterface::SignalingState new_state)
{
	SignalingChangeCallback(new_state, peerConnectionID);
}

void mzPeerConnectionObserver::OnAddTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver, const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>& streams)
{
	AddTrackCallback(receiver, streams, peerConnectionID);
}

void mzPeerConnectionObserver::OnRemoveTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver)
{
	RemoveTrackCallback(receiver, peerConnectionID);
}

void mzPeerConnectionObserver::OnDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> data_channel)
{
	DataChannelCallback(data_channel, peerConnectionID);
}

void mzPeerConnectionObserver::OnRenegotiationNeeded()
{
	RenegotiationNeededCallback(peerConnectionID);
}

void mzPeerConnectionObserver::OnIceConnectionChange(webrtc::PeerConnectionInterface::IceConnectionState new_state)
{
	IceConnectionChangeCallback(new_state, peerConnectionID);
}

void mzPeerConnectionObserver::OnIceGatheringChange(webrtc::PeerConnectionInterface::IceGatheringState new_state)
{
	IceGatheringChangeCallback(new_state, peerConnectionID);
}

void mzPeerConnectionObserver::OnIceCandidate(const webrtc::IceCandidateInterface* candidate)
{
	IceCandidateCallback(candidate, peerConnectionID);
}

void mzPeerConnectionObserver::SetSignalingChangeCallback(std::function<void(webrtc::PeerConnectionInterface::SignalingState, int)> callback)
{
	SignalingChangeCallback = callback;
}

void mzPeerConnectionObserver::SetAddTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>&, int)> callback)
{
	AddTrackCallback = callback;
}

void mzPeerConnectionObserver::SetRemoveTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, int)> callback)
{
	RemoveTrackCallback = callback;
}

void mzPeerConnectionObserver::SetDataChannelCallback(std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>, int)> callback)
{
	DataChannelCallback = callback;
}

void mzPeerConnectionObserver::SetRenegotiationCallback(std::function<void(int)> callback)
{
	RenegotiationNeededCallback = callback;
}

void mzPeerConnectionObserver::SetICEConnectionChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceConnectionState, int)> callback)
{
	IceConnectionChangeCallback = callback;
}

void mzPeerConnectionObserver::SetICEGatheringChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceGatheringState, int)> callback)
{
	IceGatheringChangeCallback = callback;
}

void mzPeerConnectionObserver::SetICECandidateCallback(std::function<void(const webrtc::IceCandidateInterface*, int)> callback)
{
	IceCandidateCallback = callback;
}
