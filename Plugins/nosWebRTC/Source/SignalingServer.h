/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once
#include "WebSocketServer.h"

/*				 ___________________________________________
				|			SIGNALING SERVER				|
				|	_________________________________		|
				|	|				|				|		|
				|	| player socket	|streamer socket|		|	
				|	|_______________|_______________|		|
				|___________________________________________|
					_____________		_____________					
					|			|		|			|
					|	player	|		|  streamer	|
					|___________|		|___________|

	We must communicate players and streamers on the same URL path (i.e. we need to conduct messages URL path-wise)
	Therefore we need to hold 
*/

class nosSignalingServer {
public:
	nosSignalingServer();
	~nosSignalingServer();

	void StartServer(int playerPort, int streamerPort);
	void Update();
	void StopServer();
	
	void SetServerCreatedCallback(const std::function<void()> callback);
	void SetServerDestroyedCallback(const std::function<void()> callback);
	void SetStreamerConnectedCallback(const std::function<void(int, std::string)> callback);
	void SetStreamerDisconnectedCallback(const std::function<void(int, std::string)> callback);
	void SetPlayerConnectedCallback(const std::function<void(int, std::string)> callback);
	void SetPlayerDisconnectedCallback(const std::function<void(int, std::string)> callback);

private:
	void OnServerCreated();
	void OnServerDestroyed();
	void OnNewStreamerConnected(int id, std::string path);
	void OnStreamerDisconnected(int id, std::string path);
	void OnNewPlayerConnected(int id, std::string path);
	void OnPlayerDisconnected(int id, std::string path);

	void OnStreamerMessage(int id, std::string path, std::string message);

	void OnPlayerMessage(int id, std::string path, std::string message);

	std::unique_ptr<nosWebSocketServer> p_ServerSocket;
	
	std::unordered_map<std::string, int> streamerPathIDMap;
	std::unordered_map<std::string, std::vector<int>> playerPathIDMap;


	std::function<void()> ServerCreatedCallback;
	std::function<void()> ServerDestroyedCallback;
	std::function<void(int, std::string)> StreamerConnectedCallback;
	std::function<void(int, std::string)> StreamerDisconnectedCallback;
	std::function<void(int, std::string)> PlayerConnectedCallback;
	std::function<void(int, std::string)> PlayerDisconnectedCallback;
	
};