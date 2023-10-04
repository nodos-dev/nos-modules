
#include "WebRTCManager.h"
#include "VideoSource.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "api/audio/audio_mixer.h"
#include "api/audio_codecs/audio_decoder_factory.h"
#include "api/audio_codecs/audio_encoder_factory.h"
#include "api/audio_codecs/builtin_audio_decoder_factory.h"
#include "api/audio_codecs/builtin_audio_encoder_factory.h"
#include "api/audio_options.h"
#include "api/create_peerconnection_factory.h"
#include "api/rtp_sender_interface.h"
#include "api/video_codecs/builtin_video_decoder_factory.h"
#include "api/video_codecs/builtin_video_encoder_factory.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "modules/audio_device/include/audio_device.h"
#include "modules/audio_processing/include/audio_processing.h"
#include "modules/video_capture/video_capture.h"
#include "modules/video_capture/video_capture_factory.h"
#include "p2p/base/port_allocator.h"
#include "pc/video_track_source.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/ref_counted_object.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "rtc_base/strings/json.h"
#include "test/vcm_capturer.h"
#include "json/json.h"
#include <MediaZ/PluginAPI.h>

bool GetStringFromJson(const Json::Value& in, std::string* out) {
    if (!in.isString()) {
        if (in.isBool()) {
            *out = rtc::ToString(in.asBool());
        }
        else if (in.isInt()) {
            *out = rtc::ToString(in.asInt());
        }
        else if (in.isUInt()) {
            *out = rtc::ToString(in.asUInt());
        }
        else if (in.isDouble()) {
            *out = rtc::ToString(in.asDouble());
        }
        else {
            return false;
        }
    }
    else {
        *out = in.asString();
    }
    return true;
}

bool GetIntFromJson(const Json::Value& in, int* out) {
    bool ret;
    if (!in.isString()) {
        ret = in.isConvertibleTo(Json::intValue);
        if (ret) {
            *out = in.asInt();
        }
    }
    else {
        long val;  // NOLINT
        const char* c_str = in.asCString();
        char* end_ptr;
        errno = 0;
        val = strtol(c_str, &end_ptr, 10);  // NOLINT
        ret = (end_ptr != c_str && *end_ptr == '\0' && !errno && val >= INT_MIN &&
            val <= INT_MAX);
        *out = val;
    }
    return ret;
}

bool GetValueFromJsonObject(const Json::Value& in,
    const std::string& k,
    Json::Value* out) {
    if (!in.isObject() || !in.isMember(k)) {
        return false;
    }

    *out = in[k];
    return true;
}

bool GetIntFromJsonObject(const Json::Value& in,
    const std::string& k,
    int* out) {
    Json::Value x;
    return GetValueFromJsonObject(in, k, &x) && GetIntFromJson(x, out);
}

bool GetStringFromJsonObject(const Json::Value& in,
    const std::string& k,
    std::string* out) {
    Json::Value x;
    return GetValueFromJsonObject(in, k, &x) && GetStringFromJson(x, out);
}

namespace {
// Names used for a IceCandidate JSON object.
const char kCandidateSdpMidName[] = "sdpMid";
const char kCandidateSdpMlineIndexName[] = "sdpMLineIndex";
const char kCandidateSdpName[] = "candidate";

// Names used for a SessionDescription JSON object.
const char kSessionDescriptionTypeName[] = "type";
const char kSessionDescriptionSdpName[] = "sdp";

class DummySetSessionDescriptionObserver
    : public webrtc::SetSessionDescriptionObserver {
public:
    static DummySetSessionDescriptionObserver* Create() {
        return new rtc::RefCountedObject<DummySetSessionDescriptionObserver>();
    }
    virtual void OnSuccess() { RTC_LOG(INFO) << __FUNCTION__; }
    virtual void OnFailure(webrtc::RTCError error) {
        RTC_LOG(INFO) << __FUNCTION__ << " " << ToString(error.type()) << ": "
            << error.message();
    }
};

}  // namespace

WebRTCManager::WebRTCManager(PeerConnectionClient* client, CustomVideoSource* customVideoSource, std::shared_ptr<AtomicQueue< std::pair<EWebRTCTasks, std::shared_ptr<void>> >> taskQueue)
    : peer_id_(-1), loopback_(false), client_(client), task_queue(nullptr){
  client_->RegisterObserver(this);
  preSetVideoSource = (customVideoSource == nullptr) ? (new CustomVideoSource()): (customVideoSource);
  task_queue = taskQueue;
}

WebRTCManager::~WebRTCManager() {
  RTC_DCHECK(!peer_connection_);
}

bool WebRTCManager::connection_active() const {
  return peer_connection_ != nullptr;
}

void WebRTCManager::SetPeerConnectedCallback(std::function<void()> callback)
{
    OnPeerConnectedCallback = callback;
}

void WebRTCManager::SetPeerDisconnectedCallback(std::function<void()> callback)
{
    OnPeerDisConnectedCallback = callback;
}

void WebRTCManager::Close() {
  client_->SignOut();
  DeletePeerConnection();
}

bool WebRTCManager::InitializePeerConnection() {
  RTC_DCHECK(!peer_connection_factory_);
  RTC_DCHECK(!peer_connection_);

  peer_connection_factory_ = webrtc::CreatePeerConnectionFactory(
      nullptr /* network_thread */, nullptr /* worker_thread */,
      nullptr /* signaling_thread */, nullptr /* default_adm */,
      webrtc::CreateBuiltinAudioEncoderFactory(),
      webrtc::CreateBuiltinAudioDecoderFactory(),
      webrtc::CreateBuiltinVideoEncoderFactory(),
      webrtc::CreateBuiltinVideoDecoderFactory(), nullptr /* audio_mixer */,
      nullptr /* audio_processing */);

  if (!peer_connection_factory_) {
    DeletePeerConnection();
    return false;
  }

  if (!CreatePeerConnection(/*dtls=*/true)) {
    DeletePeerConnection();
  }

  AddTracks();

  return peer_connection_ != nullptr;
}

bool WebRTCManager::ReinitializePeerConnectionForLoopback() {
  loopback_ = true;
  std::vector<rtc::scoped_refptr<webrtc::RtpSenderInterface>> senders =
      peer_connection_->GetSenders();
  peer_connection_ = nullptr;
  // Loopback is only possible if encryption is disabled.
  webrtc::PeerConnectionFactoryInterface::Options options;
  options.disable_encryption = true;
  peer_connection_factory_->SetOptions(options);
  if (CreatePeerConnection(false)) {
    for (const auto& sender : senders) {
      peer_connection_->AddTrack(sender->track(), sender->stream_ids());
    }
    peer_connection_->CreateOffer(
        this, webrtc::PeerConnectionInterface::RTCOfferAnswerOptions());
  }
  options.disable_encryption = false;
  peer_connection_factory_->SetOptions(options);
  return peer_connection_ != nullptr;
}

bool WebRTCManager::CreatePeerConnection(bool dtls) {
    RTC_DCHECK(peer_connection_factory_);
    RTC_DCHECK(!peer_connection_);

    webrtc::PeerConnectionInterface::RTCConfiguration config;
    config.sdp_semantics = webrtc::SdpSemantics::kUnifiedPlan;
    config.enable_dtls_srtp = dtls;
    webrtc::PeerConnectionInterface::IceServer server;
    server.uri = GetPeerConnectionString();
    config.servers.push_back(server);

    peer_connection_ = peer_connection_factory_->CreatePeerConnection(
        config, nullptr, nullptr, this);
    return peer_connection_ != nullptr;
}

void WebRTCManager::DeletePeerConnection() {
  peer_connection_ = nullptr;
  peer_connection_factory_ = nullptr;
  peer_id_ = -1;
  loopback_ = false;
}

//
// PeerConnectionObserver implementation.
//

void WebRTCManager::OnAddTrack(
    rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver,
    const std::vector<rtc::scoped_refptr<webrtc::MediaStreamInterface>>&
        streams) {
  RTC_LOG(LS_INFO) << __FUNCTION__ << " " << receiver->id();

}

void WebRTCManager::OnRemoveTrack(
    rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver) {
  RTC_LOG(LS_INFO) << __FUNCTION__ << " " << receiver->id();
}

void WebRTCManager::OnIceCandidate(const webrtc::IceCandidateInterface* candidate) {
  RTC_LOG(LS_INFO) << __FUNCTION__ << " " << candidate->sdp_mline_index();
  // For loopback test. To save some connecting delay.
  if (loopback_) {
    if (!peer_connection_->AddIceCandidate(candidate)) {
      RTC_LOG(LS_WARNING) << "Failed to apply the received candidate";
    }
    return;
  }

  Json::Value jmessage;
  jmessage[kCandidateSdpMidName] = candidate->sdp_mid();
  jmessage[kCandidateSdpMlineIndexName] = candidate->sdp_mline_index();
  std::string sdp;
  if (!candidate->ToString(&sdp)) {
    RTC_LOG(LS_ERROR) << "Failed to serialize candidate";
    return;
  }
  jmessage[kCandidateSdpName] = sdp;

  Json::StreamWriterBuilder factory;
  SendMessage(std::make_shared<std::string>(Json::writeString(factory, jmessage)));
}

//
// PeerConnectionClientObserver implementation.
//

void WebRTCManager::OnSignedIn() {
  RTC_LOG(LS_INFO) << __FUNCTION__;
  mzEngine.LogI("WebRTC Client connected to server");
}

void WebRTCManager::OnDisconnected() {
  RTC_LOG(LS_INFO) << __FUNCTION__;

  DeletePeerConnection();
}

void WebRTCManager::OnPeerConnected(int id, const std::string& name) {
  RTC_LOG(LS_INFO) << __FUNCTION__;
  mzEngine.LogI("Sucessfully connected to peer ", name);
  OnPeerConnectedCallback();
}

void WebRTCManager::OnPeerDisconnected(int id) {
  RTC_LOG(LS_INFO) << __FUNCTION__;
  if (id == peer_id_) {
    RTC_LOG(LS_INFO) << "Our peer disconnected";
     task_queue->push({EWebRTCTasks::eDISCONNECT, nullptr});
     OnPeerDisConnectedCallback();
  }
}

void WebRTCManager::OnMessageFromPeer(int peer_id, const std::string& message) {
  RTC_DCHECK(peer_id_ == peer_id || peer_id_ == -1);
  RTC_DCHECK(!message.empty());

  if (!peer_connection_.get()) {
    RTC_DCHECK(peer_id_ == -1);
    peer_id_ = peer_id;

    if (!InitializePeerConnection()) {
      RTC_LOG(LS_ERROR) << "Failed to initialize our PeerConnection instance";
      client_->SignOut();
      return;
    }
  } 
  else if (peer_id != peer_id_) {
    RTC_DCHECK(peer_id_ != -1);
    RTC_LOG(LS_WARNING)
        << "Received a message from unknown peer while already in a "
           "conversation with a different peer.";
    return;
  }

  Json::CharReaderBuilder factory;
  std::unique_ptr<Json::CharReader> reader =
      absl::WrapUnique(factory.newCharReader());
  Json::Value jmessage;
  if (!reader->parse(message.data(), message.data() + message.length(),
                     &jmessage, nullptr)) {
    RTC_LOG(LS_WARNING) << "Received unknown message. " << message;
    return;
  }
  std::string type_str;
  std::string json_object;

  GetStringFromJsonObject(jmessage, kSessionDescriptionTypeName,
                               &type_str);
  if (!type_str.empty()) {
    if (type_str == "offer-loopback") {
      // This is a loopback call.
      // Recreate the peerconnection with DTLS disabled.
      if (!ReinitializePeerConnectionForLoopback()) {
        RTC_LOG(LS_ERROR) << "Failed to initialize our PeerConnection instance";
        DeletePeerConnection();
        client_->SignOut();
      }
      return;
    }
    absl::optional<webrtc::SdpType> type_maybe =
        webrtc::SdpTypeFromString(type_str);
    if (!type_maybe) {
      RTC_LOG(LS_ERROR) << "Unknown SDP type: " << type_str;
      return;
    }
    webrtc::SdpType type = *type_maybe;
    std::string sdp;
    if (!GetStringFromJsonObject(jmessage, kSessionDescriptionSdpName,
                                      &sdp)) {
      RTC_LOG(LS_WARNING)
          << "Can't parse received session description message.";
      return;
    }
    webrtc::SdpParseError error;
    std::unique_ptr<webrtc::SessionDescriptionInterface> session_description =
        webrtc::CreateSessionDescription(type, sdp, &error);
    if (!session_description) {
      RTC_LOG(LS_WARNING)
          << "Can't parse received session description message. "
             "SdpParseError was: "
          << error.description;
      return;
    }
    RTC_LOG(LS_INFO) << " Received session description :" << message;
    peer_connection_->SetRemoteDescription(
        DummySetSessionDescriptionObserver::Create(),
        session_description.release());
    if (type == webrtc::SdpType::kOffer) {
      peer_connection_->CreateAnswer(
          this, webrtc::PeerConnectionInterface::RTCOfferAnswerOptions());
    }
  } else {
    std::string sdp_mid;
    int sdp_mlineindex = 0;
    std::string sdp;
    if (!GetStringFromJsonObject(jmessage, kCandidateSdpMidName,
                                      &sdp_mid) ||
        !GetIntFromJsonObject(jmessage, kCandidateSdpMlineIndexName,
                                   &sdp_mlineindex) ||
        !GetStringFromJsonObject(jmessage, kCandidateSdpName, &sdp)) {
      RTC_LOG(LS_WARNING) << "Can't parse received message.";
      return;
    }
    webrtc::SdpParseError error;
    std::unique_ptr<webrtc::IceCandidateInterface> candidate(
        webrtc::CreateIceCandidate(sdp_mid, sdp_mlineindex, sdp, &error));
    if (!candidate.get()) {
      RTC_LOG(LS_WARNING) << "Can't parse received candidate message. "
                             "SdpParseError was: "
                          << error.description;
      return;
    }
    if (!peer_connection_->AddIceCandidate(candidate.get())) {
      RTC_LOG(LS_WARNING) << "Failed to apply the received candidate";
      return;
    }
    RTC_LOG(LS_INFO) << " Received candidate :" << message;
  }
}

void WebRTCManager::OnMessageSent(int err) {
  // Process the next pending message if any.
  task_queue->push({EWebRTCTasks::eSEND_MESSAGE_TO_PEER,nullptr});
}

void WebRTCManager::OnServerConnectionFailure() {
    mzEngine.LogE("WebRTC Client failed to connect server!");
}

//
// MainWndCallback implementation.
//

void WebRTCManager::StartLogin(const std::string& server, int port) {
  if (client_->is_connected())
    return;
  m_server = server;
  client_->Connect(server, port, GetPeerName());
}

void WebRTCManager::DisconnectFromServer() {
  if (client_->is_connected())
    client_->SignOut();
}

void WebRTCManager::ConnectToPeer(int peer_id) {
  RTC_DCHECK(peer_id_ == -1);
  RTC_DCHECK(peer_id != -1);

  if (peer_connection_.get()) {
    return;
  }

  if (InitializePeerConnection()) {
    peer_id_ = peer_id;
    peer_connection_->CreateOffer(
        this, webrtc::PeerConnectionInterface::RTCOfferAnswerOptions());
  }
}

void WebRTCManager::AddTracks() {
  if (!peer_connection_->GetSenders().empty()) {
    return;  // Already added tracks.
  }

  rtc::scoped_refptr<webrtc::AudioTrackInterface> audio_track(
      peer_connection_factory_->CreateAudioTrack(
          kAudioLabel,
          peer_connection_factory_->CreateAudioSource(cricket::AudioOptions())
              .get()));
  auto result_or_error = peer_connection_->AddTrack(audio_track, {kStreamId});
  if (!result_or_error.ok()) {
    RTC_LOG(LS_ERROR) << "Failed to add audio track to PeerConnection: "
                      << result_or_error.error().message();
  }
  rtc::scoped_refptr<CustomVideoSource> NewVideoSource(preSetVideoSource);
  if (NewVideoSource) {
    rtc::scoped_refptr<webrtc::VideoTrackInterface> video_track_(
        peer_connection_factory_->CreateVideoTrack(kVideoLabel,NewVideoSource));

    result_or_error = peer_connection_->AddTrack(video_track_, {kStreamId});
    if (!result_or_error.ok()) {
      RTC_LOG(LS_ERROR) << "Failed to add video track to PeerConnection: "
                        << result_or_error.error().message();
    }
  } else {
    RTC_LOG(LS_ERROR) << "OpenVideoCaptureDevice failed";
  }
}

void WebRTCManager::DisconnectFromCurrentPeer() {
  RTC_LOG(LS_INFO) << __FUNCTION__;
  if (peer_connection_.get()) {
    client_->SendHangUp(peer_id_);
    DeletePeerConnection();
  }
}

void WebRTCManager::OnSuccess(webrtc::SessionDescriptionInterface* desc) {
    peer_connection_->SetLocalDescription(
        DummySetSessionDescriptionObserver::Create(), desc);

    std::string sdp;
    desc->ToString(&sdp);

    // For loopback test. To save some connecting delay.
    if (loopback_) {
        // Replace message type from "offer" to "answer"
        std::unique_ptr<webrtc::SessionDescriptionInterface> session_description =
            webrtc::CreateSessionDescription(webrtc::SdpType::kAnswer, sdp);
        peer_connection_->SetRemoteDescription(
            DummySetSessionDescriptionObserver::Create(),
            session_description.release());
        return;
    }

    Json::StyledWriter writer;
    Json::Value jmessage;
    jmessage[kSessionDescriptionTypeName] =
        webrtc::SdpTypeToString(desc->GetType());
    jmessage[kSessionDescriptionSdpName] = sdp;
    SendMessage(std::make_shared<std::string>(writer.write(jmessage)));
}

void WebRTCManager::OnFailure(webrtc::RTCError error) {
  RTC_LOG(LS_ERROR) << ToString(error.type()) << ": " << error.message();
}

void WebRTCManager::SendMessage(std::shared_ptr<std::string> json_object) {
  task_queue->push({EWebRTCTasks::eSEND_MESSAGE_TO_PEER, json_object });
}

bool WebRTCManager::MainLoop() {
  EWebRTCTasks currentTask;
  std::shared_ptr<void> data;
  bool ret = true;
  if (task_queue && task_queue->size()) {
    currentTask = task_queue->front().first;
    data = task_queue->front().second;
    task_queue->pop();
    switch (currentTask) {
      case EWebRTCTasks::eLOGIN: {
        
          if (data) {
            std::string server_port = *static_cast<std::string*>(data.get());
            size_t delimeter = server_port.find(":");
            if (delimeter != std::string::npos) {
                std::string server = server_port.substr(0, delimeter);
                //type safety is not an issue for conversion here since the MediaZ only accepts integer inputs for `port`
                int port = std::stoi(server_port.substr(delimeter + 1));
                //check for valid port numbers, 2^16/2 = 2^15 is highest valid port number
                if (port > 0 && port < (2 << 15)) {
                    StartLogin(server, port);
                    break;
                }
            }
        }

        //fallback to defaults
        StartLogin(std::string("localhost"), 8888);
        break;
      }


      case EWebRTCTasks::eCONNECT: {
        if (data) {
            int peer_id = *static_cast<int*>(data.get());
            for (const auto& [_id, _name] : client_->peers()) {
                if (peer_id == _id) {
                    ConnectToPeer(peer_id);
                    break;
                }
            }
        }
        break;
      }


      case EWebRTCTasks::eSEND_MESSAGE_TO_PEER: {
        RTC_LOG(LS_INFO) << "SEND_MESSAGE_TO_PEER";
        std::shared_ptr<std::string> msg = std::static_pointer_cast<std::string>(data);
        if (msg) {
          // For convenience, we always run the message through the queue.
          // This way we can be sure that messages are sent to the server
          // in the same order they were signaled without much hassle.
          pending_messages_.push_back(msg);
        }
        if (!pending_messages_.empty() && !client_->IsSendingMessage()) {
          msg = pending_messages_.front();
          pending_messages_.pop_front();

          if (!client_->SendToPeer(peer_id_, msg) && peer_id_ != -1) {
            RTC_LOG(LS_ERROR) << "SendToPeer failed";
            DisconnectFromServer();
          }
        }

        if (!peer_connection_.get())
          peer_id_ = -1;
        break;
      }
        

      case EWebRTCTasks::eDISCONNECT: {
        Close();
        DisconnectFromServer();
        ret = false;
        break;
      }

    }
  }
  return ret;
}


