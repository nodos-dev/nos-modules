#pragma once
#include <mutex>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "api/media_stream_interface.h"
#include "api/peer_connection_interface.h"
#include "mzCustomVideoSource.h"
#include "mzWebRTCClient.h"
#include "mzCreateSDPObserver.h"
#include "mzSetSDPObserver.h"
#include "mzPeerConnectionObserver.h"
#include "mzEncodeImageObserver.h"
#include "mzCustomVideoSink.h"

typedef rtc::scoped_refptr<webrtc::PeerConnectionInterface> PeerConnectionPtr;

class mzWebRTCManager {
public:

    mzWebRTCManager(mzWebRTCClient* p_mzWebRTCClient);
    ~mzWebRTCManager();

    bool MainLoop(int cms=0);

    void SendOffer(int id);
    void UpdateBitrates(int bitrateKBPS);
    void AddVideoSource(rtc::scoped_refptr<mzCustomVideoSource> source);
    void AddVideoSink(rtc::scoped_refptr<mzCustomVideoSink> sink);

    void SetImageEncodeCompletedCallback(std::function<void()> callback);
    void SetPeerConnectedCallback(std::function<void()> callback);
    void SetPeerDisconnectedCallback(std::function<void()> callback);
    void SetServerConnectionSuccesfulCallback(std::function<void()> callback);
    void SetServerConnectionFailedCallback(std::function<void()> callback);
    void Dispose();

    void AddRef() const;
    rtc::RefCountReleaseStatus Release() const;
protected:

    bool AddPeerConnection();
    void RemovePeerConnection(int id);

    void OnImageEncoded();

    #pragma region mzWebRTCClient Region

    void RegisterToWebRTCClientCallbacks();
    void OnServerConnectionSuccesful();
    void OnServerConnectionError();
    void OnServerConnectionClosed();
    void OnRawMessageReceived(void* data, size_t length);
    void OnSDPOfferReceived(std::string&& offer);
    void OnSDPAnswerReceived(std::string&& answer);
    void OnICECandidateReceived(std::string&& iceCandidate);

    #pragma endregion

    #pragma region mzPeerConnectionObserver Region
   
    void RegisterToPeerConnectionObserverCallbacks(mzPeerConnectionObserver* observer);
    void OnSignalingChange( webrtc::PeerConnectionInterface::SignalingState new_state, int id);
    void OnAddTrack( rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver,
        const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>& streams, int id);
    void OnRemoveTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver, int id);
    void OnDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> channel, int id);
    void OnRenegotiationNeeded(int id);
    void OnIceConnectionChange(webrtc::PeerConnectionInterface::IceConnectionState new_state, int id);
    void OnIceGatheringChange(webrtc::PeerConnectionInterface::IceGatheringState new_state, int id);
    void OnIceCandidate(const webrtc::IceCandidateInterface* candidate, int id);

    #pragma endregion
    
    #pragma region mzCreateSDPObserver Region
    void RegisterToCreateSDPObserverCallbacks(mzCreateSDPObserver* observer);
    // Will be called when SDP creation is succesful.
    void OnSDPCreateSuccess(webrtc::SessionDescriptionInterface* desc, int id);
    void OnSDPCreateFailure(webrtc::RTCError error, int id);
    #pragma endregion

    #pragma region mzSetSDPObserver Region
    void RegisterToSetSDPObserverCallbacks(mzSetSDPObserver* observer);
    //Will be called when SDP set to PeerConnection succesfully
    void OnSDPSetSuccess(int id);
    void OnSDPSetFailure(webrtc::RTCError error, int id);
    #pragma endregion
    // Send a message to the signaling server.

    bool ReadInternalIDFromPeerID(int& internalID, int peerID);
    
    std::unique_ptr<rtc::Thread> SignalingThread;
    std::unique_ptr<rtc::Thread> WorkerThread;
    //We will register ourselves so that we will be notified whether SDP creation succeed or failed
    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> p_PeerConnectionFactory;
    std::vector<std::shared_ptr<mzPeerConnectionObserver>> p_PeerConnectionObservers;
    std::vector<rtc::scoped_refptr<mzCreateSDPObserver>> p_CreateSDPObservers;
    std::vector<rtc::scoped_refptr<mzSetSDPObserver>> p_SetSDPObservers;
    std::vector<PeerConnectionPtr> p_PeerConnections;
    
    std::map<int, int> PeerConnectionIdx_PeerID;
    std::vector<rtc::scoped_refptr<mzCustomVideoSource>> p_VideoSources;
    std::vector<rtc::scoped_refptr<mzCustomVideoSink>> p_VideoSinks;
    std::vector<rtc::scoped_refptr<webrtc::VideoTrackInterface>> p_VideoTracks;

    std::unique_ptr<mzEncodeImageObserver> p_encodeObserver;
    mzWebRTCClient* p_mzWebRTCClient;
    
    
    int targetKbps = 5000;
    bool IsDisposed = false;

    std::function<void()> ImageEncodeCompletedCallback;
    std::function<void()> PeerConnectedCallback;
    std::function<void()> PeerDisconnectedCallback;
    std::function<void()> ServerConnectionSuccesfulCallback;
    std::function<void()> ServerConnectionFailedCallback;
    mutable webrtc::webrtc_impl::RefCounter ref_count_{ 0 };

    friend mzPeerConnectionObserver;
    friend mzCreateSDPObserver;
    friend mzSetSDPObserver;

};
