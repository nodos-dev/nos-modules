#include "mzSignalingServer.h"
#include "mzWebRTCJsonConfig.h"
#include "nlohmann/json.hpp"
#include <iostream>
using json = nlohmann::json;
mzSignalingServer::mzSignalingServer()
{
	//p_streamerSocket.reset(new mzWebSocketServer(1919));
	//p_playerSocket.reset(new mzWebSocketServer(7070));
}
mzSignalingServer::~mzSignalingServer()
{
	p_ServerSocket.reset();
}
void mzSignalingServer::StartServer(int streamerPort, int playerPort)
{
	p_ServerSocket.reset(new mzWebSocketServer({ streamerPort,playerPort }));

	p_ServerSocket->SetClientConnectedCallback(streamerPort, [this](int id, std::string path) {this->OnNewStreamerConnected(id, path); });
	p_ServerSocket->SetClientConnectedCallback(playerPort, [this](int id, std::string path) {this->OnNewPlayerConnected(id, path); });

	p_ServerSocket->SetClientDisconnectedCallback(streamerPort, [this](int id, std::string path) {this->OnStreamerDisconnected(id, path); });
	p_ServerSocket->SetClientDisconnectedCallback(playerPort, [this](int id, std::string path) {this->OnPlayerDisconnected(id, path); });

	p_ServerSocket->SetServerCreatedCallback([this]() {this->OnServerCreated(); });
	p_ServerSocket->SetServerDestroyedCallback([this]() {this->OnServerDestroyed(); });
}

void mzSignalingServer::Update()
{
	if (p_ServerSocket) {
		p_ServerSocket->Update();
	}
}

void mzSignalingServer::StopServer()
{
	p_ServerSocket.reset();
}

void mzSignalingServer::SetServerCreatedCallback(const std::function<void()> callback)
{
	ServerCreatedCallback = callback;
}

void mzSignalingServer::SetServerDestroyedCallback(const std::function<void()> callback)
{
	ServerDestroyedCallback = callback;
}

void mzSignalingServer::SetStreamerConnectedCallback(std::function<void(int, std::string)> callback)
{
	StreamerConnectedCallback = callback;
}

void mzSignalingServer::SetStreamerDisconnectedCallback(std::function<void(int, std::string)> callback)
{
	StreamerDisconnectedCallback = callback;
}

void mzSignalingServer::SetPlayerConnectedCallback(std::function<void(int, std::string)> callback)
{
	PlayerConnectedCallback = callback;
}

void mzSignalingServer::SetPlayerDisconnectedCallback(std::function<void(int, std::string)> callback)
{
	PlayerDisconnectedCallback = callback;
}

void mzSignalingServer::OnServerCreated()
{
	if(ServerCreatedCallback)
		ServerCreatedCallback();
}

void mzSignalingServer::OnServerDestroyed()
{
	if(ServerDestroyedCallback)
		ServerDestroyedCallback();
}

void mzSignalingServer::OnNewStreamerConnected(int id, std::string path)
{
	if (streamerPathIDMap.contains(path)) {
		//TODO: give warning to user because we will override old server on the same path from now on
	}
	streamerPathIDMap[path] = id;
	if (StreamerConnectedCallback) {
		StreamerConnectedCallback(id, path);
	}
	p_ServerSocket->SetNotifyOnClientMessage(id, [this, thePath = std::move(path), id](std::string&& message) {this->OnStreamerMessage(id, thePath, std::move(message)); });
}

void mzSignalingServer::OnStreamerDisconnected(int id, std::string path)
{
	if (streamerPathIDMap.contains(path)) {
		streamerPathIDMap.erase(path);
	}
	if (StreamerDisconnectedCallback) {
		StreamerDisconnectedCallback(id, path);
	}
}

void mzSignalingServer::OnNewPlayerConnected(int id, std::string path)
{
	playerPathIDMap[path].push_back(id);
	if (PlayerConnectedCallback) {
		PlayerConnectedCallback(id, path);
	}
	p_ServerSocket->SetNotifyOnClientMessage(id, [this, thePath = std::move(path), id](std::string&& message) {this->OnPlayerMessage(id, thePath, std::move(message)); });
}

void mzSignalingServer::OnPlayerDisconnected(int id, std::string path)
{
	if (playerPathIDMap.contains(path)) {
		auto it = std::find(playerPathIDMap[path].begin(), playerPathIDMap[path].end(), id);
		if (it != playerPathIDMap[path].end()) {
			playerPathIDMap[path].erase(it);
		}
	}
	if (PlayerDisconnectedCallback) {
		PlayerDisconnectedCallback(id, path);
	}
}

//TODO: we may ensure the message is in correct format
//but this is an internal server so additional work may not worth for now
void mzSignalingServer::OnStreamerMessage(int id, std::string path, std::string message)
{
	//First look for any player on the same path with the streamer
	if (playerPathIDMap.contains(path)) {
		json jsonMessage;
		try {

			jsonMessage = json::parse(message);
		}
		catch (std::exception e) {
			std::cerr << "Failed to parse streamer message: " << e.what() << std::endl;
			return;
		}
		int playerID = std::stoi(jsonMessage[mzWebRTCJsonConfig::peerIDKey].get<std::string>());
		//Retrieve the playerId field of json
		jsonMessage[mzWebRTCJsonConfig::peerIDKey] = std::to_string(id);


		//Check if the playerID in the same path 
		if (std::find(playerPathIDMap[path].begin(), playerPathIDMap[path].end(), playerID) != playerPathIDMap[path].end()) {
			//Just forward the message 
			p_ServerSocket->SendMessageTo(playerID, std::move(jsonMessage.dump()));
		}
		//else ? 
	}
}

void mzSignalingServer::OnPlayerMessage(int id, std::string path, std::string message)
{
	//First look if there is a streamer on that path
	if (streamerPathIDMap.contains(path)) {
		json jsonMessage;
		try {

			jsonMessage = json::parse(message);
		}
		catch (std::exception e) {
			std::cerr << "Failed to parse streamer message: " << e.what() << std::endl;
			return;
		}
		jsonMessage[mzWebRTCJsonConfig::peerIDKey] = std::to_string(id);
		p_ServerSocket->SendMessageTo(streamerPathIDMap[path], std::move(jsonMessage.dump()));
	}
}
