#pragma once
#include <memory>
#include <string>
#include "libwebsockets.h"
#include <functional>
#include <queue>
#include <thread>
typedef struct lws WebSocketInstance;
static int mz_ws_callback(struct lws* WSI, enum lws_callback_reasons reason, void* user, void* in, size_t len);

class nosWebSocketClient {
public:
	nosWebSocketClient(const std::string fullIP);
	nosWebSocketClient(const std::string server,const int port, const std::string path);

	~nosWebSocketClient();

	nosWebSocketClient(const nosWebSocketClient&) = delete;
	nosWebSocketClient& operator=(const nosWebSocketClient&) = delete;

	void PushData(std::string&& Data);
	std::string GetReceivedDataAsString();
	void Update();

	//Define callback registerers & callbacks
	void SetConnectionErrorCallback(const std::function<void()>& connectionErr);
	void SetRawMessageReceivedCallback(const std::function<void(void*, size_t)>& messageReceived);
	void SetConnectionSuccesfulCallback(const std::function<void()>& connectionSuccesful);
	void SetConnectionClosedCallback(const std::function<void()>& connectionClosed);
	
private:
	std::queue<std::string> sendQueue;
	std::queue<std::string> receivedQueue;
	struct lws_context* pContext;
	std::string serverAddres;
	std::string path;
	int port;
	lws_protocols Protocols[3];
	std::atomic<bool> shouldUpdate = true;
	void StartWebSocket();
	//This method is for ultimate sending.
	void Send();
	void ProcessReceivedData(void* data, size_t lenght);

	std::tuple<std::string, int, std::string> ResolveAddres(std::string fullAddress);
	std::function<void()> ConnectionErrorCallback;
	std::function<void(void*, size_t)> MessageReceivedCallback;
	std::function<void()> ConnectionSuccesfulCallback;
	std::function<void()> ConnectionClosedCallback;

	WebSocketInstance* pWSI;

	friend int mz_ws_callback(struct lws* WSI, enum lws_callback_reasons reason, void* user, void* in, size_t len);
};  