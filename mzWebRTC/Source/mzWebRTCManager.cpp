#include <stddef.h>
#include <iostream>
#include <stdint.h>
#include <memory>
#include <utility>
#include <vector>
#include <absl/memory/memory.h>
#include <absl/types/optional.h>
#include <api/audio/audio_mixer.h>
#include <api/audio_codecs/audio_decoder_factory.h>
#include <api/audio_codecs/audio_encoder_factory.h>
#include <api/audio_codecs/builtin_audio_decoder_factory.h>
#include <api/audio_codecs/builtin_audio_encoder_factory.h>
#include <api/audio_options.h>
#include <api/create_peerconnection_factory.h>
#include <api/rtp_sender_interface.h>
#include <api/video_codecs/builtin_video_decoder_factory.h>
#include <api/video_codecs/builtin_video_encoder_factory.h>
#include <api/video_codecs/video_decoder_factory.h>
#include <api/video_codecs/video_encoder_factory.h>
#include <modules/audio_device/include/audio_device.h>
#include <modules/audio_processing/include/audio_processing.h>
#include <modules/video_capture/video_capture.h>
#include <modules/video_capture/video_capture_factory.h>
#include <p2p/base/port_allocator.h>
#include <pc/video_track_source.h>
#include <rtc_base/checks.h>
#include <rtc_base/logging.h>
#include <rtc_base/ref_counted_object.h>
#include <rtc_base/rtc_certificate_generator.h>
#include <rtc_base/strings/json.h>
#include <test/vcm_capturer.h>
#include <nlohmann/json.hpp>
#include "mzWebRTCManager.h"
#include "mzCustomVideoSource.h"
#include "mzWebRTCJsonConfig.h"
#include "mzVideoEncoderFactory.h"

using json = nlohmann::json;

mzWebRTCManager::mzWebRTCManager(mzWebRTCClient* client) :p_mzWebRTCClient(client)
{
    //TODO: switch to factory pattern in future for checking 
    //if this is succesful or not:
    if (!p_PeerConnectionFactory) {
        if (!SignalingThread) {
            SignalingThread = rtc::Thread::Create();
            SignalingThread->SetName("mzWebRTCSignalingThread",nullptr);
            SignalingThread->Start();
        }
        if (!WorkerThread) {
            WorkerThread = rtc::Thread::Create();
            WorkerThread->SetName("mzWebRTCWorkerThread", nullptr);
            WorkerThread->Start();
        }
        p_encodeObserver = std::make_unique<mzEncodeImageObserver>([this]() {this->OnImageEncoded(); });
        auto  a =webrtc::CreateBuiltinVideoEncoderFactory();
        p_PeerConnectionFactory = webrtc::CreatePeerConnectionFactory(
            nullptr /* network_thread */, WorkerThread.get() /* worker_thread */,
            SignalingThread.get() /* signaling_thread */, nullptr /* default_adm */,
            webrtc::CreateBuiltinAudioEncoderFactory(),
            webrtc::CreateBuiltinAudioDecoderFactory(),
            std::make_unique<mzVideoEncoderFactory>(p_encodeObserver.get()),
            webrtc::CreateBuiltinVideoDecoderFactory(), nullptr /* audio_mixer */,
            nullptr /* audio_processing */);
    }


    RegisterToWebRTCClientCallbacks();
}

mzWebRTCManager::~mzWebRTCManager()
{
    Dispose();
}

bool mzWebRTCManager::MainLoop(int cms)
{
    /*if (SignalingThread) {
        SignalingThread->ProcessMessages(cms);
    }
    if (WorkerThread) {
        WorkerThread->ProcessMessages(cms);
    }*/
    return true;
}


bool mzWebRTCManager::AddPeerConnection() {
    
    webrtc::PeerConnectionInterface::RTCConfiguration config;
    config.sdp_semantics = webrtc::SdpSemantics::kUnifiedPlan;
    config.enable_dtls_srtp = true;
    webrtc::PeerConnectionInterface::IceServer server;
    server.uri = "stun:stun.l.google.com:19302"; //Use google STUN server for now.
    config.servers.push_back(server);

    int lastIdx = p_PeerConnections.size();

    p_PeerConnectionObservers.push_back(std::make_shared<mzPeerConnectionObserver>(lastIdx));
    RegisterToPeerConnectionObserverCallbacks(p_PeerConnectionObservers[lastIdx].get());

    p_PeerConnections.push_back(
        p_PeerConnectionFactory->CreatePeerConnection(config, nullptr, nullptr, p_PeerConnectionObservers[lastIdx].get()));

    if (p_PeerConnections[p_PeerConnections.size() - 1] == nullptr)
        return false;
    
    //Add video tracks
    for (auto track : p_VideoTracks) {
        auto err = p_PeerConnections[lastIdx]->AddTrack(track, {"mediaZStreamID"} );
        //TODO: handle errors?
    }

    return true;
}

void mzWebRTCManager::RemovePeerConnection(int id)
{
    if (p_PeerConnections.empty() || !PeerConnectionIdx_PeerID.contains(id))
        return;

    for (const auto& sender : p_PeerConnections[id]->GetSenders()) {
        for (auto& videoTrack : p_VideoTracks) {
            if (videoTrack == sender->track()) {
                p_PeerConnections[id]->RemoveTrack(sender);
            }
        }
    }
    //First remove the peer id from map to prevent infinite recursion
    PeerConnectionIdx_PeerID.erase(id);
    p_PeerConnections[id]->Close();
    p_PeerConnections[id] = nullptr;
    p_PeerConnections.erase(p_PeerConnections.begin() + id);

    p_PeerConnectionObservers.erase(p_PeerConnectionObservers.begin() + id);
    p_CreateSDPObservers.erase(p_CreateSDPObservers.begin() + id);
    p_SetSDPObservers.erase(p_SetSDPObservers.begin() + id);
}

void mzWebRTCManager::Dispose()
{
    if (IsDisposed)
        return;

    for (auto& peerConnection : p_PeerConnections) {

        for (const auto& sender : peerConnection->GetSenders()) {
            for (auto& videoTrack : p_VideoTracks) {
                if (videoTrack == sender->track()) {
                    peerConnection->RemoveTrack(sender);
                    videoTrack->set_enabled(false);
                }
            }
        }
        peerConnection->Close();
        peerConnection = nullptr;
    }
    p_PeerConnections.clear();

    //scoped_refptr overloads = operator so that it releases 
    // old pointer and decrements refCount when you assign new one
    for (auto& videoTrack : p_VideoTracks) {
        videoTrack = nullptr;
    }
    p_VideoTracks.clear();

    for (auto& videoSource : p_VideoSources) {
        videoSource = nullptr;
    }

    p_VideoSources.clear();
    p_PeerConnectionFactory = nullptr;

    SignalingThread->Stop();
    WorkerThread->Stop();

    p_encodeObserver.reset();
    p_mzWebRTCClient = nullptr;
    PeerConnectionIdx_PeerID.clear();

    IsDisposed = true;
}

void mzWebRTCManager::OnImageEncoded()
{
    ImageEncodeCompletedCallback();
}

void mzWebRTCManager::SendOffer(int id)
{
    if (AddPeerConnection()) {
        int internalID = p_PeerConnections.size() - 1;

        //This is crucial to handle multiple peers. We need to map peerIDs with the PeerConnection objects we created
        PeerConnectionIdx_PeerID.insert({ internalID, id });

        //TODO: may be make a function for this purpose:
        p_SetSDPObservers.push_back(new mzSetSDPObserver(internalID));
        RegisterToSetSDPObserverCallbacks(p_SetSDPObservers[internalID].get());

        p_CreateSDPObservers.push_back(new mzCreateSDPObserver(internalID));
        RegisterToCreateSDPObserverCallbacks(p_CreateSDPObservers[internalID].get());
        webrtc::PeerConnectionInterface::RTCOfferAnswerOptions offerOptions;
        offerOptions.offer_to_receive_video = 1;

        p_PeerConnections[internalID]->CreateOffer(
            p_CreateSDPObservers[internalID].get(), offerOptions);
    }
}

void mzWebRTCManager::UpdateBitrates(int bitrateKBPS)
{
    targetKbps = bitrateKBPS;
    webrtc::BitrateSettings bitrateSettings;
    bitrateSettings.max_bitrate_bps = bitrateKBPS * 10 * 1000;
    bitrateSettings.min_bitrate_bps = bitrateKBPS * 5 * 1000;
    bitrateSettings.start_bitrate_bps = bitrateKBPS * 5 * 1000;


    for (auto& peerConnection : p_PeerConnections) {
        for (auto& sender : peerConnection->GetSenders()) {
            auto bitrate = sender->GetParameters();
            for (auto& encoding : bitrate.encodings) {
                encoding.max_framerate = 300;
                encoding.max_bitrate_bps = bitrateKBPS*10*1000;
                encoding.min_bitrate_bps = bitrateKBPS*5*1000;
            }
            sender->SetParameters(bitrate);
        }
        peerConnection->SetBitrate(std::move(bitrateSettings));
    }
}

void mzWebRTCManager::AddVideoSource(rtc::scoped_refptr<mzCustomVideoSource> source)
{
    if (!p_PeerConnectionFactory)
        return;
    p_VideoSources.push_back(source);
    p_VideoTracks.push_back(p_PeerConnectionFactory->CreateVideoTrack(kVideoLabel, source));

}

void mzWebRTCManager::AddVideoSink(rtc::scoped_refptr<mzCustomVideoSink> sink)
{
    p_VideoSinks.push_back(sink);
}

void mzWebRTCManager::SetImageEncodeCompletedCallback(std::function<void()> callback)
{
    ImageEncodeCompletedCallback = callback;
}

void mzWebRTCManager::SetPeerConnectedCallback(std::function<void()> callback)
{
    PeerConnectedCallback = callback;
}

void mzWebRTCManager::SetPeerDisconnectedCallback(std::function<void()> callback)
{
    PeerDisconnectedCallback = callback;
}

void mzWebRTCManager::SetServerConnectionSuccesfulCallback(std::function<void()> callback)
{
    ServerConnectionSuccesfulCallback = callback;
}

void mzWebRTCManager::SetServerConnectionFailedCallback(std::function<void()> callback)
{
    ServerConnectionFailedCallback = callback;
}

void mzWebRTCManager::AddRef() const
{
    ref_count_.IncRef();
}

rtc::RefCountReleaseStatus mzWebRTCManager::Release() const
{
    const auto status = ref_count_.DecRef();
    if (status == rtc::RefCountReleaseStatus::kDroppedLastRef) {
        delete this;
    }
    return status;
}

#pragma region CreateSDPObserver Implementation
void mzWebRTCManager::RegisterToCreateSDPObserverCallbacks(mzCreateSDPObserver* observer)
{
    observer->SetSuccessCallback([this](auto desc, int id) {this->OnSDPCreateSuccess(desc, id); });
    observer->SetFailureCallback([this](auto error, int id) {this->OnSDPCreateFailure(error, id); });
}

void mzWebRTCManager::OnSDPCreateSuccess(webrtc::SessionDescriptionInterface* desc, int id)
{
    [[unlikely]]
    if (p_PeerConnections.empty())
    {
        //FATAL, smth is seriously wrong. Should abort?
        //abort();
        return;
    }
    p_PeerConnections[id]->SetLocalDescription(p_SetSDPObservers[id].get(), desc);

    //TODO: there might be an issue if the encryption is disabled, so you may need to implement a loopback mechanism
    std::string SDP;
    desc->ToString(&SDP);
    json jsonSDP;
    size_t bwdth = SDP.find("b=AS:30");
    if (bwdth != std::string::npos) {
        std::string newbw = "b=AS:10000";
        SDP.replace(bwdth, 7, newbw);
    }

    jsonSDP[mzWebRTCJsonConfig::typeKey] = webrtc::SdpTypeToString(desc->GetType());
    jsonSDP[mzWebRTCJsonConfig::sdpKey] = SDP;
    jsonSDP[mzWebRTCJsonConfig::peerIDKey] = std::to_string(PeerConnectionIdx_PeerID[id]);
    
    p_mzWebRTCClient->SendMessageToServer(jsonSDP.dump());


    if (p_PeerConnections[id]->GetSenders().size() > 0) {
        auto bitrate = p_PeerConnections[id]->GetSenders()[0]->GetParameters();
        for (auto& encoding : bitrate.encodings) {
            encoding.max_framerate = 300;
            encoding.max_bitrate_bps = targetKbps * 10 * 1000;
            encoding.min_bitrate_bps = targetKbps * 5 * 1000;
        }
        p_PeerConnections[id]->GetSenders()[0]->SetParameters(bitrate);
    }

}

void mzWebRTCManager::OnSDPCreateFailure(webrtc::RTCError error, int id)
{
    RTC_LOG(LS_ERROR) << ToString(error.type()) << ": " << error.message();
}
#pragma endregion

void mzWebRTCManager::RegisterToSetSDPObserverCallbacks(mzSetSDPObserver* observer)
{
    observer->SetSuccessCallback([this](int id) {OnSDPSetSuccess(id); });
    observer->SetFailureCallback([this](auto error,int id) {OnSDPSetFailure(error, id); });
}

#pragma region SetSDP Callbacks
void mzWebRTCManager::OnSDPSetSuccess(int id)
{
    // no problem
}

void mzWebRTCManager::OnSDPSetFailure(webrtc::RTCError error, int id)
{
    //TODO: should handle some stuff for SDPSetFailure
}

bool mzWebRTCManager::ReadInternalIDFromPeerID(int& internalID, int peerID)
{
    for (auto [_internalID, _peerID] : PeerConnectionIdx_PeerID) {
        if (_peerID == peerID) {
            internalID = _internalID;
            return true;
        }
    }
    return false;
}
#pragma endregion


#pragma region mzWebRTCClient Implementation
void mzWebRTCManager::RegisterToWebRTCClientCallbacks()
{
    if (!p_mzWebRTCClient)
        return;

    p_mzWebRTCClient->SetConnectionSuccesfulCallback([this]() {this->OnServerConnectionSuccesful(); });
    p_mzWebRTCClient->SetConnectionErrorCallback([this]() {this->OnServerConnectionError(); });
    p_mzWebRTCClient->SetConnectionClosedCallback([this]() {this->OnServerConnectionClosed(); });
    p_mzWebRTCClient->SetRawMessageReceivedCallback([this](void* data, size_t length) {this->OnRawMessageReceived(data, length); });
    p_mzWebRTCClient->SetSDPOfferReceivedCallback([this](std::string&& message) {this->OnSDPOfferReceived(std::move(message)); });
    p_mzWebRTCClient->SetSDPAnswerReceivedCallback([this](std::string&& message) {this->OnSDPAnswerReceived(std::move(message)); });
    p_mzWebRTCClient->SetICECandidateReceivedCallback([this](std::string&& message) {this->OnICECandidateReceived(std::move(message)); });
}

void mzWebRTCManager::OnServerConnectionSuccesful()
{
    if (ServerConnectionSuccesfulCallback) {

        ServerConnectionSuccesfulCallback();
    }
}

void mzWebRTCManager::OnServerConnectionError()
{
    if (ServerConnectionFailedCallback) {
        Dispose();
        ServerConnectionFailedCallback();
    }
}

void mzWebRTCManager::OnServerConnectionClosed()
{
    if (ServerConnectionFailedCallback) {
        Dispose();
        ServerConnectionFailedCallback();
    }
}

void mzWebRTCManager::OnRawMessageReceived(void* data, size_t length)
{
}

void mzWebRTCManager::OnSDPOfferReceived(std::string&& offer)
{
    //
    if (AddPeerConnection()) {
        json jsonOffer = json::parse(offer);
        {
            std::string peerIDstr = jsonOffer[mzWebRTCJsonConfig::peerIDKey];
            int peerID = std::stoi(peerIDstr);
            int internalID = p_PeerConnections.size() - 1;
            
            //This is crucial to handle multiple peers. We need to map peerIDs with the PeerConnection objects we created
            PeerConnectionIdx_PeerID.insert({ internalID, peerID });
            
            webrtc::SdpParseError error;
            
            std::unique_ptr<webrtc::SessionDescriptionInterface> webRTCSessionDescription =
                webrtc::CreateSessionDescription(webrtc::SdpType::kOffer, jsonOffer[mzWebRTCJsonConfig::sdpKey].get<std::string>(), &error);

            if (!webRTCSessionDescription) {
                //abort the mission. Clear the recently created objects in AddPeerConnection and remove them from map.
                p_PeerConnections.pop_back();
                p_PeerConnectionObservers.pop_back();
                p_CreateSDPObservers.pop_back();
                p_SetSDPObservers.pop_back();
                PeerConnectionIdx_PeerID.erase(internalID);
                std::cout << "SDP parsing failed: " << error.description << std::endl;
                return;
            }

            //TODO: may be make a function for this purpose:
            p_SetSDPObservers.push_back(new mzSetSDPObserver(internalID));
            RegisterToSetSDPObserverCallbacks(p_SetSDPObservers[internalID].get());

            p_PeerConnections[internalID]->SetRemoteDescription(
                p_SetSDPObservers[internalID].get(), webRTCSessionDescription.release());

            p_CreateSDPObservers.push_back(new mzCreateSDPObserver(internalID));
            RegisterToCreateSDPObserverCallbacks(p_CreateSDPObservers[internalID].get());

            p_PeerConnections[internalID]->CreateAnswer(
                p_CreateSDPObservers[internalID].get(), webrtc::PeerConnectionInterface::RTCOfferAnswerOptions());

        }
    }
}

void mzWebRTCManager::OnSDPAnswerReceived(std::string&& answer)
{
    //Get the peerID from the answer and use the corresponding PeerConnection for the rest
    json jsonAnswer = json::parse(answer);
    {
        std::string peerIDstr = jsonAnswer[mzWebRTCJsonConfig::peerIDKey];
        int peerID = std::stoi(peerIDstr);
        int internalID = -1;
        if(!ReadInternalIDFromPeerID(internalID, peerID)){
            //oops..
            std::cerr << "No corresponding peer connection found for the answer from peer: " << peerID;
            return;
        }


        webrtc::SdpParseError error;

        std::unique_ptr<webrtc::SessionDescriptionInterface> webRTCSessionDescription =
            webrtc::CreateSessionDescription(webrtc::SdpType::kAnswer, jsonAnswer[mzWebRTCJsonConfig::sdpKey].get<std::string>(), &error);

        if (!webRTCSessionDescription) {
            //Abort the mission, clear corresponding items
            p_PeerConnections.erase(p_PeerConnections.begin() + internalID);
            p_PeerConnectionObservers.erase(p_PeerConnectionObservers.begin() + internalID);
            p_CreateSDPObservers.erase(p_CreateSDPObservers.begin() + internalID);
            p_SetSDPObservers.erase(p_SetSDPObservers.begin() + internalID);
            PeerConnectionIdx_PeerID.erase(internalID);
            std::cout << "SDP parsing failed: " << error.description << std::endl;
            return;
        }

        //Forward answer to webrtc internal stuff
        p_PeerConnections[internalID]->SetRemoteDescription(
            p_SetSDPObservers[internalID].get(), webRTCSessionDescription.release());


    }
}

void mzWebRTCManager::OnICECandidateReceived(std::string&& iceCandidate)
{
    json jsonICE = json::parse(iceCandidate);
    int peerID = std::stoi(jsonICE[mzWebRTCJsonConfig::peerIDKey].get<std::string>());
    int internalID;
    if (!ReadInternalIDFromPeerID(internalID, peerID)) {
        std::cerr << "No corresponding peer connection found for the ICE candidate from peer: " << peerID;
        return;
    }
    int sdpMidLineIdx = jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::sdpMidLineIndexKey].get<int>();
    std::string sdpMid = jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::sdpMidKey].get<std::string>();
    std::string candidateStr = jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::candidateKey].get<std::string>();
    webrtc::SdpParseError error;

    std::unique_ptr<webrtc::IceCandidateInterface> candidate(webrtc::CreateIceCandidate(sdpMid, sdpMidLineIdx, candidateStr, &error));
    if (!candidate) {
        std::cerr << "Received candidate message parse failed: " << error.description;
        return;
    }
    p_PeerConnections[internalID]->AddIceCandidate(candidate.get());
    //check for error when adding?
}
#pragma endregion

#pragma region PeerConnectionObserver Implementation
void mzWebRTCManager::RegisterToPeerConnectionObserverCallbacks(mzPeerConnectionObserver* observer)
{
    observer->SetSignalingChangeCallback(
        [this](auto new_state, int id) {this->OnSignalingChange(new_state, id); });
    observer->SetAddTrackCallback(
        [this](auto receiver, auto& streams, int id) {OnAddTrack(receiver, streams, id); });
    observer->SetRemoveTrackCallback(
        [this](auto receiver, int id) {this->OnRemoveTrack(receiver, id); });
    observer->SetDataChannelCallback(
        [this](auto channel, int id) {this->OnDataChannel(channel, id); });
    observer->SetRenegotiationCallback(
        [this](int id) {this->OnRenegotiationNeeded(id); });
    observer->SetICEConnectionChangeCallback(
        [this](auto new_state, int id) {this->OnIceConnectionChange(new_state, id); });
    observer->SetICEGatheringChangeCallback(
        [this](auto new_state, int id) {this->OnIceGatheringChange(new_state, id); });
    observer->SetICECandidateCallback(
        [this](auto candidate, int id) {this->OnIceCandidate(candidate, id); });

}

void mzWebRTCManager::OnSignalingChange(webrtc::PeerConnectionInterface::SignalingState new_state, int id)
{
}

void mzWebRTCManager::OnAddTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver, const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>& streams, int id)
{
    if (receiver->track()->kind() == webrtc::MediaStreamTrackInterface::kVideoKind) {
        auto track = receiver->track().release();
        auto* videoTrack = static_cast<webrtc::VideoTrackInterface*>(track);
        for (const auto& videoSink : p_VideoSinks) {
            if (videoSink->IsAvailable) {
                videoTrack->AddOrUpdateSink(videoSink, rtc::VideoSinkWants());
                videoSink->IsAvailable = false;
            }
        }
        receiver->SetJitterBufferMinimumDelay(0.05);
    }
}

void mzWebRTCManager::OnRemoveTrack(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver, int id)
{
}

void mzWebRTCManager::OnDataChannel(rtc::scoped_refptr<webrtc::DataChannelInterface> channel, int id)
{
}

void mzWebRTCManager::OnRenegotiationNeeded(int id)
{
}

void mzWebRTCManager::OnIceConnectionChange(webrtc::PeerConnectionInterface::IceConnectionState new_state, int id)
{
    if (new_state == webrtc::PeerConnectionInterface::IceConnectionState::kIceConnectionConnected
        && PeerConnectedCallback) {
        PeerConnectedCallback();
    }
    else if (new_state == webrtc::PeerConnectionInterface::kIceConnectionFailed 
        && PeerDisconnectedCallback) {
        //Handle the errors!!1
        //TODO: delete peer connections
        PeerDisconnectedCallback();
        RemovePeerConnection(id);
    }
    else if (new_state == webrtc::PeerConnectionInterface::kIceConnectionClosed 
        && PeerDisconnectedCallback) {
        //PeerDisconnectedCallback();
        //RemovePeerConnection(id);
    }
}

void mzWebRTCManager::OnIceGatheringChange(webrtc::PeerConnectionInterface::IceGatheringState new_state, int id)
{
}

void mzWebRTCManager::OnIceCandidate(const webrtc::IceCandidateInterface* candidate, int id)
{
    //This callback indicates that ice candidates for US is ready and we should send it to the server.
    /*
    * Here is the expected structure:
     {
        "type": "iceCandidate",
        "playerId": "101",
        "candidate": {
            "candidate": "candidate:884439561 1 udp 2122260223 172.30.32.1 59614 typ host generation 0 ufrag q2YZ network-id 1",
            "sdpMLineIndex": 0,
            "sdpMid": "0"
         }
      }
    
    */
    
    std::string ICEcandidate;
    candidate->ToString(&ICEcandidate);
    json jsonICE;
    
    jsonICE[mzWebRTCJsonConfig::peerIDKey] = std::to_string(PeerConnectionIdx_PeerID[id]);
    jsonICE[mzWebRTCJsonConfig::typeKey] = mzWebRTCJsonConfig::typeICE;
    jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::sdpMidKey] = candidate->sdp_mid();
    jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::sdpMidLineIndexKey] = candidate->sdp_mline_index();
    jsonICE[mzWebRTCJsonConfig::candidateKey][mzWebRTCJsonConfig::candidateKey] = ICEcandidate;
    p_mzWebRTCClient->SendMessageToServer(jsonICE.dump());
}
#pragma endregion