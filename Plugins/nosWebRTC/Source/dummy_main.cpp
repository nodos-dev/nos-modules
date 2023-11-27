#include "WebSocketClient.h"
#include "WebSocketServer.h"
#include "WebRTCManager.h"
#include "rtc_base/ssl_adapter.h"
#include "rtc_base/string_utils.h"  // For ToUtf8
#include "rtc_base/win32_socket_init.h"
#include "rtc_base/win32_socket_init.h"
#include "rtc_base/win32_socket_server.h"
#include <iostream>
#include "SignalingServer.h"
nosSignalingServer* server;
int id = -1;

void PlayerConnected(int id, std::string path) {
	std::cout << "Player "<<id<<" connected to : " << path << std::endl;
}

void PlayerDisconnected(int id, std::string path) {
	std::cout << "Player " << id << " disconnected from : " << path << std::endl;
}

void StreamerConnected(int id, std::string path) {
	std::cout << "Streamer "<<id<<" connected to : " << path << std::endl;
}
void StreamerDisconnected(int id, std::string path) {
	std::cout << "Streamer " << id << " disconnected from : " << path << std::endl;
}

int main() {
	server = new nosSignalingServer();
	server->SetPlayerConnectedCallback(PlayerConnected);
	server->SetPlayerDisconnectedCallback(PlayerDisconnected);
	server->SetStreamerConnectedCallback(StreamerConnected);
	server->SetStreamerDisconnectedCallback(StreamerDisconnected);
	server->StartServer(80, 8888);
	
	//client.SendMessageToServer("{\"type\":\"config\",\"peerConnectionOptions\":{}}");
	std::string received;
	while (true) {
		server->Update();
		int a = 5;
	}
	return 0;
}