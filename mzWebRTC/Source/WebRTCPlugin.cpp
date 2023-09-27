#include <MediaZ/PluginAPI.h>
#include <Builtins_generated.h>
#include <MediaZ/Helpers.hpp>
#include <AppService_generated.h>
#include <mzUtil/Thread.h>

#include <Windows.h>
#include <shellapi.h>  // must come after windows.h

#include <string>
#include <vector>

#include "peer_connection_client.h"
#include "rtc_base/checks.h"
#include "rtc_base/ssl_adapter.h"
#include "rtc_base/string_utils.h"  // For ToUtf8
#include "rtc_base/win32_socket_init.h"
#include "system_wrappers/include/field_trial.h"
#include "test/field_trial.h"
#include "WebRTCManager.h"
#include "VideoSource.h"
#include <memory>

// clang-format off
// clang formating would change include order.
#include <windows.h>
#include <shellapi.h>  // must come after windows.h
// clang-format on

#include <string>
#include <vector>

#include "rtc_base/checks.h"
#include "rtc_base/constructor_magic.h"
#include "rtc_base/ssl_adapter.h"
#include "rtc_base/string_utils.h"  // For ToUtf8
#include "rtc_base/win32_socket_init.h"
#include "rtc_base/win32_socket_server.h"
#include "system_wrappers/include/field_trial.h"
#include "test/field_trial.h"
#include "WebRTCManager.h"
// mzNodes

MZ_INIT();
MZ_REGISTER_NAME(In)
MZ_REGISTER_NAME(ServerIP)
MZ_REGISTER_NAME(Port)
MZ_REGISTER_NAME(Out);
MZ_REGISTER_NAME(WebRTCClient);




/*
Here is the list of current tasks for WebRTC
Usage of SEND_MESSAGE_TO_PEER highly discouraged unless you really know what you are doing
//TODO: Do not forget to update this comment section of EWebRTCTasks if you update the enum in future
enum EWebRTCTasks {
	eLOGIN,
	eCONNECT,
	eSEND_MESSAGE_TO_PEER,
	eDISCONNECT
};
*/


//The interface between medaiZ and WebRTC, stores the task qeueue and launches the connection thread

struct mzWebRTCInterface {
public:
	mzWebRTCInterface() {
		webrtcTaskQueue = std::make_shared< AtomicQueue< std::pair<EWebRTCTasks, void*> >>();
	}
	~mzWebRTCInterface() {
		if(RTCThread.joinable())
			RTCThread.join();
	}
	/// <summary>
	/// This will automatically pus eLOGIN task, do not push it again to queue before/after
	/// </summary>
	void StartConnection() {
		webrtcTaskQueue->push({ EWebRTCTasks::eLOGIN,nullptr });
		RTCThread = std::thread([this]() {StartRTCThread(webrtcTaskQueue); });
	}

	void PushTask(EWebRTCTasks task, void* data) {
		webrtcTaskQueue->push({task,data});
	}

private:
	std::shared_ptr<AtomicQueue< std::pair<EWebRTCTasks, void*> >> webrtcTaskQueue;
	std::thread RTCThread;
	void StartRTCThread(std::shared_ptr<AtomicQueue< std::pair<EWebRTCTasks, void*> >> theQueue) {
		rtc::WinsockInitializer winsock_init;
		rtc::Win32SocketServer w32_ss;
		rtc::Win32Thread w32_thread(&w32_ss);
		rtc::ThreadManager::Instance()->SetCurrentThread(&w32_thread);
		// InitFieldTrialsFromString stores the char*, so the char array must outlive
		// the application.
		std::shared_ptr<CustomVideoSource> myVideoSource(new CustomVideoSource());
		// CustomVideoSource* theVS = new CustomVideoSource();
		rtc::InitializeSSL();

		PeerConnectionClient client;
		CustomVideoSource* thisVS = new CustomVideoSource();

		rtc::scoped_refptr<WebRTCManager> manager(
			new rtc::RefCountedObject<WebRTCManager>(&client, thisVS, theQueue));


		while (manager.get()->MainLoop()) {
			w32_thread.ProcessMessages(1);
		}

		rtc::CleanupSSL();
	}
};

mzWebRTCInterface mzWebRTC;

struct WebRTCNodeContext : mz::NodeContext {

	
	mzResourceShareInfo InputCached;
	mzResourceShareInfo TempOutput = {};
	//On Node Created
	WebRTCNodeContext(mz::fb::Node const* node) :NodeContext(node){
	}

	~WebRTCNodeContext() override {
	}

	void  OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {
		
		if (pinName == MZN_In) {

			InputCached = mz::DeserializeTextureInfo(value->Data);
			mzEngine.LogI("Input value on CNN changed!");
		}
	}

	virtual mzResult ExecuteNode(const mzNodeExecuteArgs* args) override {
		auto values = mz::GetPinValues(args);


		//if (Worker.isDone) {
		//	Worker.isNewFrameReady = true;
		//	mzResourceShareInfo out = mz::DeserializeTextureInfo(values[MZN_Out]);;
		//	mzResourceShareInfo tmp = out;
		//	mzEngine.ImageLoad(Worker.data,
		//		mzVec2u(out.Info.Texture.Width, out.Info.Texture.Height),
		//		MZ_FORMAT_R8G8B8A8_SRGB, &tmp);
		//	{
		//		mzCmd cmd;
		//		mzEngine.Begin(&cmd);
		//		mzEngine.Copy(cmd, &tmp, &out, 0);
		//		mzEngine.End(cmd);
		//		mzEngine.Destroy(&tmp);
		//	}
		//	//return MZ_RESULT_SUCCESS;
		//}

		//return MZ_RESULT_FAILED;
		return MZ_RESULT_SUCCESS;
	}

	mzResult BeginCopyTo(mzCopyInfo* cpy) override {
		//cpy->ShouldCopyTexture = true;
		//cpy->CopyTextureFrom = InputCached;
		//cpy->CopyTextureTo = Worker.InputRGBA8;
		//cpy->ShouldSubmitAndWait = true;
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 1;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC("ConnectToServer");
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			auto values = mz::GetPinValues(nodeArgs);
			mzWebRTC.StartConnection();
			};

		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders) {
		*outCount = 0;
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes) {
		*count = 0;
		if (!passes)
			return MZ_RESULT_SUCCESS;

		return MZ_RESULT_SUCCESS;
	}



};

extern "C"
{

	MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, mzNodeFunctions** outFunctions) {
		*outCount = (size_t)(1);
		if (!outFunctions)
			return MZ_RESULT_SUCCESS;

		MZ_BIND_NODE_CLASS(MZN_WebRTCClient, WebRTCNodeContext, outFunctions[0]);


		return MZ_RESULT_SUCCESS;
	}
}

