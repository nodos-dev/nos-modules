#ifndef MZ_WEBRTC_JSON_CONFIG
#define MZ_WEBRTC_JSON_CONFIG

#include <string>
namespace mzWebRTCJsonConfig{
	static std::string candidateKey = "candidate";
	static std::string sdpKey = "sdp";
	static std::string sdpMidKey = "sdpMid";
	static std::string sdpMidLineIndexKey = "sdpMLineIndex";
	static std::string peerIDKey = "playerId";
	static std::string typeKey = "type";
	static std::string typeAnswer = "answer";
	static std::string typeOffer = "offer";
	static std::string typeICE = "iceCandidate";
}

#endif // !MZ_WEBRTC_JSON_CONFIG