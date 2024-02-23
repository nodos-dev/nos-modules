// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include "WebRTCClient.h"
#include "WebRTCJsonConfig.h"

using json = nlohmann::json;

nosWebRTCClient::nosWebRTCClient() :
	currentState(EClientState::eNOT_CONNECTED), p_nosWebSocketClient(nullptr), clientID(-1)
{
	clientName = "nosWebRTCClient_" +  std::to_string(rand());
}

nosWebRTCClient::nosWebRTCClient(std::string name) : 
	currentState(EClientState::eNOT_CONNECTED), clientName(name), p_nosWebSocketClient(nullptr) , clientID(-1)
{
}

void nosWebRTCClient::ConnectToServer(std::string fullAddres)
{
	p_nosWebSocketClient.reset(new nosWebSocketClient(fullAddres));
	RegisterWebSocketCallbacks();
}

EClientState nosWebRTCClient::GetCurrentState() const
{
	return currentState;
}

const Peers& nosWebRTCClient::GetPeers() const
{
	return peers;
}

void nosWebRTCClient::SendMessageToServer(std::string&& message)
{
 	p_nosWebSocketClient->PushData(std::move(message));
}

const int nosWebRTCClient::GetID() const
{
	return clientID;
}

void nosWebRTCClient::Update()
{
	if (p_nosWebSocketClient) {
		p_nosWebSocketClient->Update();
	}
}

void nosWebRTCClient::OnConnectionSuccesful()
{
	currentState = eCONNECTED;

	if (ConnectionSuccesfulCallback)
		ConnectionSuccesfulCallback();
}

void nosWebRTCClient::OnMessageReceived(void* data, size_t length)
{
	InterpretReceivedMessage();
	MessageReceivedCallback(data, length);
}

void nosWebRTCClient::OnConnectionError()
{
	currentState = eNOT_CONNECTED;
	
	if(ConnectionErrorCallback)
		ConnectionErrorCallback();
}

void nosWebRTCClient::OnConnectionClosed()
{
	currentState = eNOT_CONNECTED;

	if(ConnectionClosedCallback)
		ConnectionClosedCallback();
}

void nosWebRTCClient::InterpretReceivedMessage()
{
	if (!p_nosWebSocketClient) {
		return;
	}

	std::string currentMessage = p_nosWebSocketClient->GetReceivedDataAsString();
	if (!currentMessage.empty()) {
		json jsonMessage = json::parse(currentMessage);
		if (jsonMessage.contains(nosWebRTCJsonConfig::typeKey) && jsonMessage.contains(nosWebRTCJsonConfig::peerIDKey)) {
			// Offers/answers has sdp key 
			if (jsonMessage.contains(nosWebRTCJsonConfig::sdpKey)) {
				if (jsonMessage[nosWebRTCJsonConfig::typeKey] == nosWebRTCJsonConfig::typeOffer && SDPOfferReceivedCallback) {
					SDPOfferReceivedCallback(std::move(currentMessage));
				}
				else if (jsonMessage[nosWebRTCJsonConfig::typeKey] == nosWebRTCJsonConfig::typeAnswer && SDPAnswerReceivedCallback) {
					SDPAnswerReceivedCallback(std::move(currentMessage));
				}
			}
			else if (jsonMessage.contains(nosWebRTCJsonConfig::candidateKey)) {
				ICECandidateReceivedCallback(std::move(currentMessage));
			}
		}
	}
}

void nosWebRTCClient::ResetConnections()
{
	peers.clear();
	currentState = EClientState::eNOT_CONNECTED;
	clientID = -1;
}

void nosWebRTCClient::RegisterWebSocketCallbacks()
{
	if (!p_nosWebSocketClient)
		return;
	p_nosWebSocketClient->SetConnectionClosedCallback([this]() {this->OnConnectionClosed(); });
	p_nosWebSocketClient->SetConnectionErrorCallback([this]() {this->OnConnectionError(); });
	p_nosWebSocketClient->SetConnectionSuccesfulCallback([this]() {this->OnConnectionSuccesful(); });
	p_nosWebSocketClient->SetRawMessageReceivedCallback([this](void* data, size_t length) {this->OnMessageReceived(data, length); });
}

#pragma region Set Callbacks

void nosWebRTCClient::SetConnectionErrorCallback(const std::function<void()> connectionErr)
{
	ConnectionErrorCallback = connectionErr;
}

void nosWebRTCClient::SetRawMessageReceivedCallback(const std::function<void(void*, size_t)> messageReceived)
{
	MessageReceivedCallback = messageReceived;
}

void nosWebRTCClient::SetConnectionSuccesfulCallback(const std::function<void()> connectionSuccesful)
{
	ConnectionSuccesfulCallback = connectionSuccesful;
}

void nosWebRTCClient::SetConnectionClosedCallback(const std::function<void()> connectionClosed)
{
	ConnectionClosedCallback = connectionClosed;
}

void nosWebRTCClient::SetSDPOfferReceivedCallback(std::function<void(std::string&&)> sdpOfferReceived)
{
	SDPOfferReceivedCallback = sdpOfferReceived;
}

void nosWebRTCClient::SetSDPAnswerReceivedCallback(std::function<void(std::string&&)> sdpAnswerReceived)
{
	SDPAnswerReceivedCallback = sdpAnswerReceived;
}

void nosWebRTCClient::SetICECandidateReceivedCallback(std::function<void(std::string&&)> iceCandidateReceived)
{
	ICECandidateReceivedCallback = iceCandidateReceived;
}

#pragma endregion

