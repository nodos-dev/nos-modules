#include "mzWebRTCClient.h"
#include "mzWebRTCJsonConfig.h"

using json = nlohmann::json;

mzWebRTCClient::mzWebRTCClient() :
	currentState(EClientState::eNOT_CONNECTED), p_mzWebSocketClient(nullptr), clientID(-1)
{
	clientName = "mzWebRTCClient_" +  std::to_string(rand());
}

mzWebRTCClient::mzWebRTCClient(std::string name) : 
	currentState(EClientState::eNOT_CONNECTED), clientName(name), p_mzWebSocketClient(nullptr) , clientID(-1)
{
}

void mzWebRTCClient::ConnectToServer(std::string fullAddres)
{
	p_mzWebSocketClient.reset(new mzWebSocketClient(fullAddres));
	RegisterWebSocketCallbacks();
}

EClientState mzWebRTCClient::GetCurrentState() const
{
	return currentState;
}

const Peers& mzWebRTCClient::GetPeers() const
{
	return peers;
}

void mzWebRTCClient::SendMessageToServer(std::string&& message)
{
 	p_mzWebSocketClient->PushData(std::move(message));
}

const int mzWebRTCClient::GetID() const
{
	return clientID;
}

void mzWebRTCClient::Update()
{
	if (p_mzWebSocketClient) {
		p_mzWebSocketClient->Update();
	}
}

void mzWebRTCClient::OnConnectionSuccesful()
{
	currentState = eCONNECTED;

	if (ConnectionSuccesfulCallback)
		ConnectionSuccesfulCallback();
}

void mzWebRTCClient::OnMessageReceived(void* data, size_t length)
{
	InterpretReceivedMessage();
	MessageReceivedCallback(data, length);
}

void mzWebRTCClient::OnConnectionError()
{
	currentState = eNOT_CONNECTED;
	
	if(ConnectionErrorCallback)
		ConnectionErrorCallback();
}

void mzWebRTCClient::OnConnectionClosed()
{
	currentState = eNOT_CONNECTED;

	if(ConnectionClosedCallback)
		ConnectionClosedCallback();
}

void mzWebRTCClient::InterpretReceivedMessage()
{
	if (!p_mzWebSocketClient) {
		return;
	}

	std::string currentMessage = p_mzWebSocketClient->GetReceivedDataAsString();
	if (!currentMessage.empty()) {
		json jsonMessage = json::parse(currentMessage);
		if (jsonMessage.contains(mzWebRTCJsonConfig::typeKey) && jsonMessage.contains(mzWebRTCJsonConfig::peerIDKey)) {
			// Offers/answers has sdp key 
			if (jsonMessage.contains(mzWebRTCJsonConfig::sdpKey)) {
				if (jsonMessage[mzWebRTCJsonConfig::typeKey] == mzWebRTCJsonConfig::typeOffer && SDPOfferReceivedCallback) {
					SDPOfferReceivedCallback(std::move(currentMessage));
				}
				else if (jsonMessage[mzWebRTCJsonConfig::typeKey] == mzWebRTCJsonConfig::typeAnswer && SDPAnswerReceivedCallback) {
					SDPAnswerReceivedCallback(std::move(currentMessage));
				}
			}
			else if (jsonMessage.contains(mzWebRTCJsonConfig::candidateKey)) {
				ICECandidateReceivedCallback(std::move(currentMessage));
			}
		}
	}
}

void mzWebRTCClient::ResetConnections()
{
	peers.clear();
	currentState = EClientState::eNOT_CONNECTED;
	clientID = -1;
}

void mzWebRTCClient::RegisterWebSocketCallbacks()
{
	if (!p_mzWebSocketClient)
		return;
	p_mzWebSocketClient->SetConnectionClosedCallback([this]() {this->OnConnectionClosed(); });
	p_mzWebSocketClient->SetConnectionErrorCallback([this]() {this->OnConnectionError(); });
	p_mzWebSocketClient->SetConnectionSuccesfulCallback([this]() {this->OnConnectionSuccesful(); });
	p_mzWebSocketClient->SetRawMessageReceivedCallback([this](void* data, size_t length) {this->OnMessageReceived(data, length); });
}

#pragma region Set Callbacks

void mzWebRTCClient::SetConnectionErrorCallback(const std::function<void()> connectionErr)
{
	ConnectionErrorCallback = connectionErr;
}

void mzWebRTCClient::SetRawMessageReceivedCallback(const std::function<void(void*, size_t)> messageReceived)
{
	MessageReceivedCallback = messageReceived;
}

void mzWebRTCClient::SetConnectionSuccesfulCallback(const std::function<void()> connectionSuccesful)
{
	ConnectionSuccesfulCallback = connectionSuccesful;
}

void mzWebRTCClient::SetConnectionClosedCallback(const std::function<void()> connectionClosed)
{
	ConnectionClosedCallback = connectionClosed;
}

void mzWebRTCClient::SetSDPOfferReceivedCallback(std::function<void(std::string&&)> sdpOfferReceived)
{
	SDPOfferReceivedCallback = sdpOfferReceived;
}

void mzWebRTCClient::SetSDPAnswerReceivedCallback(std::function<void(std::string&&)> sdpAnswerReceived)
{
	SDPAnswerReceivedCallback = sdpAnswerReceived;
}

void mzWebRTCClient::SetICECandidateReceivedCallback(std::function<void(std::string&&)> iceCandidateReceived)
{
	ICECandidateReceivedCallback = iceCandidateReceived;
}

#pragma endregion

