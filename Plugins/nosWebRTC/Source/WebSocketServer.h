/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once
#include <memory>
#include <string>
#include "libwebsockets.h"
#include <functional>
#include <queue>
#include <thread>
typedef struct lws WebSocketInstance;
static int mz_ws_server_callback(struct lws* WSI, enum lws_callback_reasons reason, void* user, void* in, size_t len);

class nosWebSocketServer {
public:
	//TODO: we may need to specify a limit for max number of connections
	nosWebSocketServer(std::vector<int>&& portNumber);
	//Set html and css files for server
	nosWebSocketServer(std::vector<int>&& portNumber, std::string httpMountOrigin, std::string defaultFilename);

	~nosWebSocketServer();

	nosWebSocketServer(const nosWebSocketServer&) = delete;
	nosWebSocketServer& operator=(const nosWebSocketServer&) = delete;

	void SendMessageTo(int clientID, std::string&& data);

	//This pops messages from the queue of corresponding client,
	//should be iterated until getting an empty string
	std::string GetMessagesFrom(int clientID);

	//Returns Client ID and the URL path client connected to
	const std::vector<std::pair<int, std::string>> GetClientInfos() const;

	void Update();

	//Define callback registerers & callbacks
	// TODO: can be improved with unregistering behaviour for memory management (i.e. what happens if a registered class dies: runtime error)
	//For all clients 
	void SetClientConnectedCallback(int port, const std::function<void(int,std::string)>& clientConnected);
	void SetClientDisconnectedCallback(int port, const std::function<void(int,std::string)>& clientDisconnected);
	//Subscribe to specific client
	void SetNotifyOnClientMessage(int clientID, const std::function<void(std::string&&)>& notifyOnClientMessage);
	void SetNotifyOnClientDisconnected(int clientID, const std::function<void()>& notifyOnClientDisconnected);

	void SetServerCreatedCallback(const std::function<void()> callback);
	void SetServerDestroyedCallback(const std::function<void()> callback);
	
private:
	//Which ports are we listening to
	std::vector<int> ports;

	//Client ID and data
	std::unordered_map<int, std::queue<std::string>> sendQueueMap;
	
	//Client ID and data
	std::unordered_map<int, std::queue<std::string>> receivedQueueMap;

	//ClientID, Client WSI, Client Path on server
	std::unordered_map<int, std::pair<WebSocketInstance*, std::string>> clientWSIMap;
	//Never decreases. We will use this to assign unique IDs
	int cumulativeConnected = 0;

	struct lws_context* pContext;
	lws_protocols Protocols[3];
	std::atomic<bool> shouldUpdate = true;
	void StartWebSocket();

	//This method is for ultimate sending
	void Send(WebSocketInstance* wsi);
	void ProcessReceivedData(WebSocketInstance* wsi, void* data, size_t length);
	void ClientDisconnected(WebSocketInstance* wsi);

	std::function<void()> ServerDestroyedCallback;
	//Client Connected Callbacks subscribed to specific port
	std::unordered_map<int,std::vector<std::function<void(int,std::string)>>> ClientConnectedCallbacks;
	//Client Disconnected Callbacks subscribed to specific port
	std::unordered_map<int, std::vector<std::function<void(int, std::string)>>> ClientDisconnectedCallbacks;

	std::unordered_map<int, std::vector<std::function<void(std::string&&)> >> NotifyOnClientMessageMap;
	std::unordered_map<int, std::vector<std::function<void()> >> NotifyOnClientDisconnectedMap;

	//Port Vhost map
	std::unordered_map<int, lws_vhost*> portVHOSTMap;

	//This is useful for serving html/css interface of the server on browser.
	//We can specify html css files and browser will use them
	lws_http_mount httpMount;

	friend int mz_ws_server_callback(struct lws* WSI, enum lws_callback_reasons reason, void* user, void* in, size_t len);
};  