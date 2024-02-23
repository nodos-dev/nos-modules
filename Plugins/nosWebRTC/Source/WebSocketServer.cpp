// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <iostream>
#include <exception>
#include "WebSocketServer.h"

nosWebSocketServer::nosWebSocketServer(std::vector<int>&& port) : ports(port)
{
	httpMount = {
		/* .mount_next */		NULL,		/* linked-list "next" */
		/* .mountpoint */		"localhost",		/* mountpoint URL */
		/* .origin */			"",  /* serve from dir */
		/* .def */				"",	/* default filename */
		/* .protocol */			NULL,
		/* .cgienv */			NULL,
		/* .extra_mimetypes */		NULL,
		/* .interpret */		NULL,
		/* .cgi_timeout */		0,
		/* .cache_max_age */		0,
		/* .auth_mask */		0,
		/* .cache_reusable */		0,
		/* .cache_revalidate */		0,
		/* .cache_intermediaries */	0,
		/* .origin_protocol */		LWSMPRO_FILE,	/* files in a dir */
		/* .mountpoint_len */		1,		/* char count */
		/* .basic_auth_login_file */	NULL,
	};
	if (ports.size() > 0) {
		//TODO: check if valid port numbers
		StartWebSocket();
	}
	else {
		throw std::exception("No ports specified for nosWebSocketServer");
	}
}

nosWebSocketServer::nosWebSocketServer(std::vector<int>&& port, std::string httpMountOrigin, std::string defaultFilename) : ports(port)
{
	//httpMountOrigin.erase(std::find(httpMountOrigin.begin(), httpMountOrigin.end(), '\0'), httpMountOrigin.end());
	//defaultFilename.erase(std::find(defaultFilename.begin(), defaultFilename.end(), '\0'), defaultFilename.end());
	httpMount = {
		/* .mount_next */		NULL,		/* linked-list "next" */
		/* .mountpoint */		"localhost",		/* mountpoint URL */
		/* .origin */			httpMountOrigin.c_str(),  /* serve from dir */
		/* .def */				defaultFilename.c_str(),	/* default filename */
		/* .protocol */			NULL,
		/* .cgienv */			NULL,
		/* .extra_mimetypes */		NULL,
		/* .interpret */		NULL,
		/* .cgi_timeout */		0,
		/* .cache_max_age */		0,
		/* .auth_mask */		0,
		/* .cache_reusable */		0,
		/* .cache_revalidate */		0,
		/* .cache_intermediaries */	0,
		/* .origin_protocol */		LWSMPRO_FILE,	/* files in a dir */
		/* .mountpoint_len */		1,		/* char count */
		/* .basic_auth_login_file */	NULL,
	};
	if (ports.size() > 0) {
		//TODO: check if valid port numbers
		StartWebSocket();
	}
	else {
		throw std::exception("No ports specified for nosWebSocketServer");
	}
}


nosWebSocketServer::~nosWebSocketServer()
{
	//Update();
 	shouldUpdate = false;
	if (pContext) {
		lws_context_destroy(pContext);
	}
	ServerDestroyedCallback();
}


void nosWebSocketServer::SendMessageTo(int clientID, std::string&& data)
{
	if (!clientWSIMap.contains(clientID))
		return;

	data = std::string(LWS_PRE, ' ') + data;
	sendQueueMap[clientID].push(std::move(data));
}

std::string nosWebSocketServer::GetMessagesFrom(int clientID)
{
	if (receivedQueueMap[clientID].empty())
		return std::string();

	std::string message = receivedQueueMap[clientID].front();
	receivedQueueMap[clientID].pop();
	return std::move(message);
}

const std::vector<std::pair<int,std::string>> nosWebSocketServer::GetClientInfos() const
{
	std::vector<std::pair<int, std::string>> clientIds;
	for (const auto& [id, InstancePath] : clientWSIMap) {
		const auto& [instance, path] = InstancePath;
		clientIds.push_back({id, path});
	}
	return clientIds;
}

//Use this in a thread
void nosWebSocketServer::Update()
{
	if (pContext) {
		for (const auto& [id, WSIPath] : clientWSIMap) {
			const auto& [wsi, path] = WSIPath;
			lws_callback_on_writable(wsi);
		}
		lws_service(pContext, 0);
	}
}

void nosWebSocketServer::SetClientConnectedCallback(int port, const std::function<void(int, std::string)>& clientConnected)
{
	ClientConnectedCallbacks[port].push_back(clientConnected);
}

void nosWebSocketServer::SetClientDisconnectedCallback(int port, const std::function<void(int, std::string)>& clientDisconnected)
{
	ClientDisconnectedCallbacks[port].push_back(clientDisconnected);
}

void nosWebSocketServer::SetNotifyOnClientMessage(int clientID, const std::function<void(std::string&&)>& notifyOnClientMessage)
{
	NotifyOnClientMessageMap[clientID].push_back(notifyOnClientMessage);
}

void nosWebSocketServer::SetNotifyOnClientDisconnected(int clientID, const std::function<void()>& notifyOnClientDisconnected)
{
	NotifyOnClientDisconnectedMap[clientID].push_back(notifyOnClientDisconnected);
}

void nosWebSocketServer::SetServerCreatedCallback(const std::function<void()> callback)
{
	//This is not ideal, but we create the lws context via constructor (for safety)
	//and hence we cant call ServerCreatedCallback anytime later; instead we will check if server exists
	//and then notify
	if (pContext)
		callback();
}

void nosWebSocketServer::SetServerDestroyedCallback(const std::function<void()> callback)
{
	ServerDestroyedCallback = callback;
}

void nosWebSocketServer::Send(WebSocketInstance* wsi)
{
	for (auto& [id, queue] : sendQueueMap) {
		
		if (!clientWSIMap.contains(id) || clientWSIMap[id].first != wsi)
			continue;

		if (queue.empty())
			return;

		std::string message = queue.front();
		int remainingDataToSent = message.size() - LWS_PRE;
		while (remainingDataToSent > 0) {
			int sent = lws_write(wsi, (unsigned char*)&message[LWS_PRE], message.size() - LWS_PRE, (lws_write_protocol)LWS_WRITE_TEXT);
			remainingDataToSent -= sent;
		}
		queue.pop();
	}
}

void nosWebSocketServer::StartWebSocket()
{
	//Library handles http handshake?
	pContext = nullptr;
	memset(&Protocols, 0, sizeof(lws_protocols) * 3);

	//Protocols refers to subprotocols, they agreed upon conventions for how the client and server will communicate.
	//This subprotocols can define the format of the messages whether data should be cmpressed or how to interpret different types of messages
	//etc. When the clien establishes a WebSocket connection it can send a list of supported subprotocols then the server selects one it supports
	//and includes this in its handshake response then the both client and server know which protocol to use for their communication.
	Protocols[0].name = "mz-ws";
	Protocols[0].callback = mz_ws_server_callback;
	Protocols[0].per_session_data_size = 0;
	Protocols[0].rx_buffer_size = 10 * 1024 * 1024; //received buffer size. Note that data can be received in fragments!

	Protocols[1].name = nullptr;
	Protocols[1].callback = nullptr;
	Protocols[1].per_session_data_size = 0;

	lws_context_creation_info Info;
	memset(&Info, 0, sizeof Info);

	Info.port = CONTEXT_PORT_NO_LISTEN;
	Info.mounts = &httpMount;
	Info.vhost_name = "localhost";
	Info.protocols = &Protocols[0];
	Info.user = this;

	Info.options |= LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;
	Info.options |= LWS_SERVER_OPTION_EXPLICIT_VHOSTS;

	pContext = lws_create_context(&Info);
	if (!pContext)
		throw std::exception("Port not found in the addres!");

	for (int i = 0; i < ports.size(); i++) {
		lws_context_creation_info vhost = { 0 };
		std::string name = std::string("localhost:" + std::to_string(ports[i]));
		vhost.vhost_name = name.c_str();
		vhost.port = ports[i];
		vhost.protocols = &Protocols[0];

		portVHOSTMap[ports[i]]= lws_create_vhost(pContext, &vhost);
	}
}

void nosWebSocketServer::ProcessReceivedData(WebSocketInstance* wsi, void* data, size_t length)
{
	if (length == 0)
		return;

	int id = -1;
	for (const auto& [_id, instance] : clientWSIMap) {
		if (instance.first == wsi) {
			id = _id;
		}
	}
	if (id == -1) {
		throw std::exception("Message received from unknown client!");
	}

	std::string strData = std::string(reinterpret_cast<char *>(data), length);
	std::cout << "Message received from id: " << id << strData << std::endl;
	if (NotifyOnClientMessageMap.contains(id)) {
		for (const auto& callback : NotifyOnClientMessageMap[id]) {
			std::string dataToSubscriber = strData;
			callback(std::move(dataToSubscriber));
		}
	}

	receivedQueueMap[id].push(std::move(strData));

}

void nosWebSocketServer::ClientDisconnected(WebSocketInstance* wsi)
{
	int id = -1;
	std::string path;
	for (const auto& [_id, InstancePath] : clientWSIMap) {
		const auto& [instance, _path] = InstancePath;
		if (instance == wsi) {
			id = _id;
			path = _path;
		}
	}

	if (id == -1) {
		throw std::exception("Unknown client disconnected!");
	}

	for (const auto& [port, callbacks] : ClientDisconnectedCallbacks) {
		
		if (portVHOSTMap[port] != lws_get_vhost(wsi))
			continue;

		for (const auto& callback : callbacks) {
			callback(id, path);
		}
	}

	if (NotifyOnClientDisconnectedMap.contains(id)) {
		for (const auto& callback : NotifyOnClientDisconnectedMap[id]) {
			callback();
		}
	}

	clientWSIMap.erase(id);
	NotifyOnClientDisconnectedMap.erase(id);
	NotifyOnClientMessageMap.erase(id);
}

static int mz_ws_server_callback(struct lws* WSI, enum lws_callback_reasons Reason, void* User, void* In, size_t Len)
{
	struct lws_context* Context = lws_get_context(WSI);
	nosWebSocketServer* Socket = (nosWebSocketServer*)lws_context_user(Context);
	Socket->ProcessReceivedData(WSI, In, Len);

	switch (Reason)
	{

		case LWS_CALLBACK_ESTABLISHED:
		{
			//Succesful connection
			/*if (Socket && Socket->pWSI == WSI && Socket->ConnectionSuccesfulCallback)
				Socket->ConnectionSuccesfulCallback();*/
			//Socket->ProcessReceivedData(WSI, In, Len);
			lws_set_timeout(WSI, NO_PENDING_TIMEOUT, 0);

			break;
		}
		case LWS_CALLBACK_SERVER_NEW_CLIENT_INSTANTIATED:
		{
			int id = (1000 + Socket->cumulativeConnected++);
			Socket->clientWSIMap[id] = { WSI,"" };

			break;
		}
		case LWS_CALLBACK_SERVER_WRITEABLE:
		{
			//Weird name but this reason tells that the client is writable indeed
			Socket->Send(WSI);
			break;
		}
		case LWS_CALLBACK_RECEIVE:
		{
			//We received a message from client
			//Socket->ProcessReceivedData(WSI, In,Len);
			lws_set_timeout(WSI, NO_PENDING_TIMEOUT, 0);
			break;
		}
		case LWS_CALLBACK_HTTP:
		{
			char buf[LWS_PRE + 256];
			lws_hdr_copy(WSI, buf, sizeof(buf), WSI_TOKEN_GET_URI);
			//Socket->ProcessReceivedData(WSI, In, Len);
			break;
		}
		case LWS_CALLBACK_FILTER_HTTP_CONNECTION:
		{
			char URL[256];
			if (lws_hdr_copy(WSI, URL, sizeof(URL), WSI_TOKEN_GET_URI)) {
				for (auto& [id, InstancePath] : Socket->clientWSIMap) {
					auto& [instance, path] = InstancePath;
					if (instance == WSI) {
						if (path.empty()) {
							path = std::string(URL);
						}
						//We should inform the subscribers only after we retrieved the path information
						//to prevent query misses on paths
						for(const auto& [port, callbacks] : Socket->ClientConnectedCallbacks) {
							if (Socket->portVHOSTMap[port] != lws_get_vhost(WSI))
								continue;
							for (const auto& callback : callbacks) {
								if (callback) {
									callback(id, path);
								}
							}
						}
					}
				}
			}
			break;
		}
		case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
		{
			char URL[256];
			if (lws_hdr_copy(WSI, URL, sizeof(URL), WSI_TOKEN_GET_URI)) {
				for (auto& [id, InstancePath] : Socket->clientWSIMap) {
					auto& [instance, path] = InstancePath;
					if (instance == WSI) {
						if (path.empty()) {
							path = std::string(URL);
						}
						//We should inform the subscribers only after we retrieved the path information
						//to prevent query misses on paths
						for (const auto& [port, callbacks] : Socket->ClientConnectedCallbacks) {
							if (Socket->portVHOSTMap[port] != lws_get_vhost(WSI))
								continue;
							for (const auto& callback : callbacks) {
								if (callback) {
									callback(id, path);
								}
							}
						}
					}
				}
			}
			break;
		}
		case LWS_CALLBACK_CLOSED:
		{
			Socket->ClientDisconnected(WSI);
			break;
		}
	}

	return 0;
}