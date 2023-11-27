#pragma once
#include <map>
#include <memory>
#include <string>
#include <rtc_base/net_helpers.h>
#include <rtc_base/physical_socket_server.h>
#include <rtc_base/third_party/sigslot/sigslot.h>
#include <nlohmann/json.hpp>
#include "WebSocketClient.h"

typedef std::map<int, std::string> Peers;

enum EClientState {
	eNOT_CONNECTED,
	eSIGNING_IN,
	eRESOLVING,
	eCONNECTED,
	SIGNING_OUT_WAITING,
	SIGNING_OUT,
};

class nosWebRTCClient : public sigslot::has_slots<> {
public:
	nosWebRTCClient();
	/// <summary>
	/// Create client with name
	/// </summary>
	/// <param name="name"></param>
	nosWebRTCClient(std::string name);
	~nosWebRTCClient() = default;

	void ConnectToServer(std::string fullAddres);
	EClientState GetCurrentState() const;
	const Peers& GetPeers() const;
	void SendMessageToServer(std::string&& message);
	const int GetID() const;

	void Update();
	
	//Register callbacks to this class.
	void SetConnectionErrorCallback(const std::function<void()> connectionErr);
	void SetRawMessageReceivedCallback(const std::function<void(void*, size_t)> messageReceived);
	void SetConnectionSuccesfulCallback(const std::function<void()> connectionSuccesful);
	void SetConnectionClosedCallback(const std::function<void()> connectionClosed);

	void SetSDPOfferReceivedCallback(std::function<void(std::string&&)> sdpOfferReceived);
	void SetSDPAnswerReceivedCallback(std::function<void(std::string&&)> sdpAnswerReceived);
	void SetICECandidateReceivedCallback(std::function<void(std::string&&)> iceCandidateReceived);

private:
	void ResetConnections();
	//Register this class' callbacks to websocket class
	void RegisterWebSocketCallbacks();
	//for re-usability. We need to re-create websocket if we want to change IP
	std::unique_ptr<nosWebSocketClient> p_nosWebSocketClient; 
	//might need to hold prev state later?
	Peers peers;
	std::atomic<EClientState> currentState; 
	std::string clientName;
	int clientID;

	//WebSocket Callbacks
	void OnConnectionSuccesful();
	void OnMessageReceived(void* data, size_t length);
	void OnConnectionError();
	void OnConnectionClosed();

	//WebRTC client will interpret the message and notify on Ice Candidate, SDP Offer/Answer
	//those who registered
	void InterpretReceivedMessage();

	//TODO: We may need to handle more than one callbacks. Consider storing them
	// using vector<> instead of one object
	std::function<void()> ConnectionErrorCallback;
	std::function<void(void*, size_t)> MessageReceivedCallback;
	std::function<void()> ConnectionSuccesfulCallback;
	std::function<void()> ConnectionClosedCallback;

	std::function<void(std::string&&)> SDPOfferReceivedCallback;
	std::function<void(std::string&&)> SDPAnswerReceivedCallback;
	std::function<void(std::string&&)> ICECandidateReceivedCallback;

};