#pragma once
#include "api/peer_connection_interface.h"

class nosPeerConnectionObserver : public webrtc::PeerConnectionObserver {
public:
	nosPeerConnectionObserver(int peerConnectionID);
    
    void SetSignalingChangeCallback(std::function<void(webrtc::PeerConnectionInterface::SignalingState, int)> callback);
    void SetAddTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>,
        const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>&, int)> callback);
    void SetRemoveTrackCallback(std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, int)> callback);
    void SetDataChannelCallback(std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>, int)> callback);
    void SetRenegotiationCallback(std::function<void(int)> callback);
    void SetICEConnectionChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceConnectionState, int)> callback);
    void SetICEGatheringChangeCallback(std::function<void(webrtc::PeerConnectionInterface::IceGatheringState, int)> callback);
    void SetICECandidateCallback(std::function<void(const webrtc::IceCandidateInterface*, int)> callback);

private:
	//This will be used for multiple peer connections and it is crucial!
	int peerConnectionID;

	// Inherited via PeerConnectionObserver
    void OnSignalingChange(webrtc::PeerConnectionInterface::SignalingState new_state) override;
    void OnAddTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver,
        const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>& streams) override;
    void OnRemoveTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver) override;
    void OnDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> channel) override;

    //This means a new offer answer exchange is required.
    //For example: when a use adds a new screen share, the apps need to renegotiate
    void OnRenegotiationNeeded() override;

    //Triggered when the state of ICE agent changes
    //Could be useful for detecting ICE connection is established/demolished/failed
    void OnIceConnectionChange(webrtc::PeerConnectionInterface::IceConnectionState new_state) override;
    void OnIceGatheringChange(webrtc::PeerConnectionInterface::IceGatheringState new_state) override;
    void OnIceCandidate(const webrtc::IceCandidateInterface* candidate) override;
    //Legacy? 
    void OnIceConnectionReceivingChange(bool receiving) override {};

    std::function<void(webrtc::PeerConnectionInterface::SignalingState, int)> SignalingChangeCallback;
    std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>,
        const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>&, int)> AddTrackCallback;
    std::function<void(rtc::scoped_refptr<webrtc::RtpReceiverInterface>, int)> RemoveTrackCallback;
    std::function<void(rtc::scoped_refptr<webrtc::DataChannelInterface>, int)> DataChannelCallback;
    std::function<void(int)> RenegotiationNeededCallback;
    std::function<void(webrtc::PeerConnectionInterface::IceConnectionState, int)> IceConnectionChangeCallback;
    std::function<void(webrtc::PeerConnectionInterface::IceGatheringState, int)> IceGatheringChangeCallback;
    std::function<void(const webrtc::IceCandidateInterface*, int)> IceCandidateCallback;
};