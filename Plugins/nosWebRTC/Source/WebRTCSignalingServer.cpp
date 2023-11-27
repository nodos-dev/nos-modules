#include <MediaZ/PluginAPI.h>
#include <Builtins_generated.h>
#include <MediaZ/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include "mzSignalingServer.h"

MZ_REGISTER_NAME(WebRTCSignalingServer);
MZ_REGISTER_NAME(StreamerPort)
MZ_REGISTER_NAME(PlayerPort)

struct WebRTCSignalingServerNodeContext : mz::NodeContext {
	mzUUID StartServerUUID;
	mzUUID StopServerUUID;

	std::thread ServerThread;
	std::mutex UpdateMutex;
	std::condition_variable ServerUpdateCV;
	std::atomic_bool ShouldUpdateServer = false;
	std::atomic_bool ShouldSuspendUpdate = false;
	std::unique_ptr<mzSignalingServer> p_server;
	int StreamerCount = 0;
	int PlayerCount = 0;
	WebRTCSignalingServerNodeContext(mz::fb::Node const* node) {
		//lws_set_log_level(LLL_ERR | LLL_WARN | LLL_INFO | LLL_DEBUG, nullptr);

		for (auto func : *node->functions()) {
			if (strcmp(func->class_name()->c_str(), "StartServer") == 0) {
				StartServerUUID = *func->id();
			}
			else if (strcmp(func->class_name()->c_str(), "StopServer") == 0) {
				StopServerUUID = *func->id();
			}
		}

		flatbuffers::FlatBufferBuilder fbb;

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &StopServerUUID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
	}

	~WebRTCSignalingServerNodeContext() {
		if(p_server)
			p_server->StopServer();
		ShouldUpdateServer = false;
		ShouldSuspendUpdate = false;
		if (ServerThread.joinable()) {
			ServerUpdateCV.notify_one();
			ServerThread.join();
		}
	}

	void RegisterCallbacks() {
		if (p_server) {
			p_server->SetStreamerConnectedCallback([this](int id, std::string path) {this->OnStreamerConnected(id, path); });
			p_server->SetStreamerDisconnectedCallback([this](int id, std::string path) {this->OnStreamerDisconnected(id, path); });
			p_server->SetPlayerConnectedCallback([this](int id, std::string path) {this->OnPlayerConnected(id, path); });
			p_server->SetPlayerDisconnectedCallback([this](int id, std::string path) {this->OnPlayerDisconnected(id, path); });
			p_server->SetServerCreatedCallback([this]() {this->OnServerCreated(); });
			p_server->SetServerDestroyedCallback([this]() {this->OnServerDestroyed(); });
		}
	}

	void OnServerCreated() {
		flatbuffers::FlatBufferBuilder fbb;

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &StartServerUUID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &StopServerUUID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));
	}

	void OnServerDestroyed() {
		flatbuffers::FlatBufferBuilder fbb;

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &StartServerUUID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &StopServerUUID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
	}

	void OnStreamerConnected(int id, std::string path) {
		StreamerCount++;
		mzEngine.LogI("Streamer %d connected to %s!", id, path);
		mzEngine.WatchLog("Connected Streamers: ", std::to_string(StreamerCount).c_str());
	}

	void OnStreamerDisconnected(int id, std::string path) {
		StreamerCount--;
		mzEngine.LogI("Streamer %d disconnected from %s!", id, path);
		mzEngine.WatchLog("Connected Streamers: ", std::to_string(StreamerCount).c_str());
	}

	void OnPlayerConnected(int id, std::string path) {
		PlayerCount++;
		mzEngine.LogI("Player %d connected to %s!", id, path);
		mzEngine.WatchLog("Connected Players: ", std::to_string(PlayerCount).c_str());
	}

	void OnPlayerDisconnected(int id, std::string path) {
		PlayerCount--;
		mzEngine.LogI("Player %d disconnected from %s!", id, path);
		mzEngine.WatchLog("Connected Players: ", std::to_string(PlayerCount).c_str());
	}

	void ServerUpdate() {
		while (ShouldUpdateServer) {
			std::unique_lock<std::mutex> serverLock(UpdateMutex);
			if (ShouldSuspendUpdate) {
				ServerUpdateCV.wait(serverLock);
			}
			if (p_server) {
				p_server->Update();
			}
		}
	}

	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 2;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC("StartServer");
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			if (WebRTCSignalingServerNodeContext* serverNode = static_cast<WebRTCSignalingServerNodeContext*>(ctx)) {
				
				std::unique_lock<std::mutex> serverLock(serverNode->UpdateMutex);

				auto values = mz::GetPinValues(nodeArgs);
				int streamerPort = *mz::GetPinValue<int>(values, MZN_StreamerPort);
				int playerPort = *mz::GetPinValue<int>(values, MZN_PlayerPort);
				if (!serverNode->p_server) {
					serverNode->p_server.reset(new mzSignalingServer());
					serverNode->RegisterCallbacks();
					serverNode->ShouldUpdateServer = true;
					serverNode->ServerThread = std::thread([serverNode]() {serverNode->ServerUpdate(); });
				}
				
				serverNode->p_server->StartServer(streamerPort, playerPort);
				serverNode->ShouldSuspendUpdate = false;
				serverNode->ServerUpdateCV.notify_one();

			}

			};

		names[1] = MZ_NAME_STATIC("StopServer");
		fns[1] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			if (WebRTCSignalingServerNodeContext* serverNode = static_cast<WebRTCSignalingServerNodeContext*>(ctx)) {
				serverNode->ShouldSuspendUpdate = true;
				auto values = mz::GetPinValues(nodeArgs);
				serverNode->p_server->StopServer();
			}

			};

		return MZ_RESULT_SUCCESS;
	}
};

void RegisterWebRTCSignalingServer(mzNodeFunctions* outFunctions) {
	MZ_BIND_NODE_CLASS(MZN_WebRTCSignalingServer, WebRTCSignalingServerNodeContext, outFunctions);
}
