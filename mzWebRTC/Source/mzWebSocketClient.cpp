#include "mzWebSocketClient.h"
#include <iostream>
#include <exception>

mzWebSocketClient::mzWebSocketClient(const std::string fullIP)
{
	try {
		auto [_server, _port, _path] = ResolveAddres(fullIP);
		serverAddres = _server;
		port = _port;
		path = _path;
		StartWebSocket();
	}
	catch (std::exception& E) {
		throw(E);
	}
}

mzWebSocketClient::mzWebSocketClient(const std::string server, const int _port, const std::string _path)
	:serverAddres(server), port(_port), path(_path)
{	
	StartWebSocket();
}

mzWebSocketClient::~mzWebSocketClient()
{
 	shouldUpdate = false;
	if (pContext) {
		lws_context_destroy(pContext);
	}
}

//Not the most efficient way but does the job now
//Also webrtc does not communicates with server very often
void mzWebSocketClient::PushData(std::string&& message)
{
	message = std::string(LWS_PRE, ' ') + message;
	sendQueue.push(std::move(message));
}

//We trust return value optimization for large datas
std::string mzWebSocketClient::GetReceivedDataAsString()
{
	if (receivedQueue.empty())
		return std::string();

	std::string tempData = receivedQueue.front();
	receivedQueue.pop();
	return std::move(tempData);
}

//Use this in a thread
void mzWebSocketClient::Update()
{
	if (pContext) {
		lws_service(pContext, 0);
		lws_callback_on_writable(pWSI);
	}
}

void mzWebSocketClient::SetConnectionErrorCallback(const std::function<void()>& connectionErr)
{
	ConnectionErrorCallback = connectionErr;
}

void mzWebSocketClient::SetRawMessageReceivedCallback(const std::function<void(void*, size_t)>& messageReceived)
{
	MessageReceivedCallback = messageReceived;
}

void mzWebSocketClient::SetConnectionSuccesfulCallback(const std::function<void()>& connectionSuccesful)
{
	ConnectionSuccesfulCallback = connectionSuccesful;
}

void mzWebSocketClient::SetConnectionClosedCallback(const std::function<void()>& connectionClosed)
{
	ConnectionClosedCallback = connectionClosed;
}

void mzWebSocketClient::Send()
{
	if (sendQueue.empty())
		return;
	std::string message = sendQueue.front();
	sendQueue.pop();
	int remainingDataToSent = message.size() - LWS_PRE;
	while (remainingDataToSent > 0) {
		int sent = lws_write(pWSI, (unsigned char*)&message[LWS_PRE], message.size() - LWS_PRE, (lws_write_protocol)LWS_WRITE_TEXT);
		remainingDataToSent -= sent;
		if (remainingDataToSent > 0) {
			std::cout << "Data partially send in mzWebSocketClient" << std::endl;
		}
	}
}

void mzWebSocketClient::StartWebSocket()
{
	//Library handles http handshake?
	pContext = nullptr;
	memset(&Protocols, 0, sizeof(lws_protocols) * 3);

	//Protocols refers to subprotocols, they agreed upon conventions for how the client and server will communicate.
	//This subprotocols can define the format of the messages whether data should be cmpressed or how to interpret different types of messages
	//etc. When the clien establishes a WebSocket connection it can send a list of supported subprotocols then the server selects one it supports
	//and includes this in its handshake response then the both client and server know which protocol to use for their communication.
	Protocols[0].name = "mz-ws";
	Protocols[0].callback = mz_ws_callback;
	Protocols[0].per_session_data_size = 0;
	Protocols[0].rx_buffer_size = 10 * 1024 * 1024; //received buffer size. Note that data can be received in fragments!

	Protocols[1].name = nullptr;
	Protocols[1].callback = nullptr;
	Protocols[1].per_session_data_size = 0;

	lws_context_creation_info Info;
	memset(&Info, 0, sizeof Info);

	Info.port = CONTEXT_PORT_NO_LISTEN;
	Info.protocols = &Protocols[0];
	Info.gid = -1;
	Info.uid = -1;
	Info.user = this;

	Info.options |= LWS_SERVER_OPTION_DISABLE_IPV6;
	pContext = lws_create_context(&Info);
	if (!pContext)
		throw std::exception("Port not found in the addres!");

	if (!path.starts_with('/')) {
		path = '/' + path;
	}
	struct lws_client_connect_info connect_info = {};

	connect_info.context = pContext;
	connect_info.address = serverAddres.c_str(); // The address of the WebSocket server
	connect_info.port = port; // Port of the WebSocket server
	connect_info.path = path.c_str(); // The path of the WebSocket resource
	connect_info.host = serverAddres.c_str(); // Set the hostname
	connect_info.origin = "mediaZ"; // Origin might be used by server for security checks to prevent possible attacks
	connect_info.protocol = Protocols[0].name; // The first protocol we've defined

	pWSI = lws_client_connect_via_info(&connect_info);
	if (!pWSI)
		throw std::exception("LWS connection failed!");

}

std::tuple<std::string, int, std::string> mzWebSocketClient::ResolveAddres(std::string fullAddress)
{
	std::string server;
	int port;
	std::string path;
	if ("ws://" == fullAddress.substr(0, 5)) {
		fullAddress = fullAddress.substr(5, fullAddress.size() - 5);
	}
	size_t portIdx = fullAddress.find(':');
	size_t pathIdx = fullAddress.find('/');
	
	if (portIdx == std::string::npos)
		throw std::invalid_argument("Invalid format, port not not found");
	
	if (pathIdx == std::string::npos)
		path = '/';
	else {
		path = fullAddress.substr(pathIdx, fullAddress.size() - pathIdx);
	}

	server = fullAddress.substr(0, portIdx);
	port = std::atoi(fullAddress.substr(portIdx + 1, (pathIdx - portIdx)).c_str());
	

	return { server, port, path };
}

//TODO: make it so that it can handle large chunks
void mzWebSocketClient::ProcessReceivedData(void* data, size_t length)
{
	//TODO: make it safer
	std::string strData = std::string(reinterpret_cast<char *>(data), length);
	receivedQueue.push(std::move(strData));
}

static int mz_ws_callback(struct lws* WSI, enum lws_callback_reasons Reason, void* User, void* In, size_t Len)
{
	struct lws_context* Context = lws_get_context(WSI);
	mzWebSocketClient* Socket = (mzWebSocketClient*)lws_context_user(Context);
	switch (Reason)
	{
		case LWS_CALLBACK_CLIENT_ESTABLISHED:
		{
			//Succesful connection
			if (Socket && Socket->pWSI == WSI && Socket->ConnectionSuccesfulCallback)
				Socket->ConnectionSuccesfulCallback();
			lws_set_timeout(WSI, NO_PENDING_TIMEOUT, 0);

		}
		break;
		case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
		{
			//Unsuccesful connection
			if (Socket && Socket->pWSI == WSI && Socket->ConnectionErrorCallback) {
				Socket->ConnectionErrorCallback();
			}
		}
		break;
		case LWS_CALLBACK_CLIENT_RECEIVE:
		{
			//We received a message from server.
			if (Socket && Socket->pWSI == WSI) {
				Socket->ProcessReceivedData(In, Len);
				if(Socket->MessageReceivedCallback)
					Socket->MessageReceivedCallback(In, Len);
			}

			lws_set_timeout(WSI, NO_PENDING_TIMEOUT, 0);
			break;
		}
		case LWS_CALLBACK_CLIENT_WRITEABLE:
		{
			//This reason indicates that the other endpoint is ready for receiving message!
			//This is important because this is WHEN we should send the data.
			//It also means our WebSocket wrapper should hold the data to be sent at some data structure!!!

			//Also do not forget to call lws_callback_on_writable() method when you want to send data to a socket as soon as 
			// the socket is ready to accept more data, i.e. writable
			// You dont call this immediately before sending data, instead you call it when you know you have a data sendable
			// and then callback will be triggered with LWS_CALLBACK_CLIENT_WRITEABLE when the other endpoint is ready!
			// 
			if (Socket && Socket->pWSI == WSI) {
				//std::cout << "Socket is writable" << std::endl;
				Socket->Send();
			}
			//TODO: Use your SEND method with lws_write here!!!
			//check(Socket->Wsi == Wsi);
			//Socket->OnRawWebSocketWritable(Wsi);
			//lws_callback_on_writable(Wsi);
			//lws_set_timeout(Wsi, NO_PENDING_TIMEOUT, 0);
			break;
		}
		case LWS_CALLBACK_CLIENT_CLOSED:
		{
			if (Socket && Socket->pWSI == WSI && Socket->ConnectionClosedCallback)
				Socket->ConnectionClosedCallback();
		}
	}

	return 0;
}