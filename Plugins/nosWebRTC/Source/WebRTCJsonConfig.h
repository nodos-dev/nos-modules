/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#ifndef NOS_WEBRTC_JSON_CONFIG
#define NOS_WEBRTC_JSON_CONFIG

#include <string>
namespace nosWebRTCJsonConfig{
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

#endif // !NOS_WEBRTC_JSON_CONFIG