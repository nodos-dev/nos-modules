#include <MediaZ/PluginAPI.h>
#include <Builtins_generated.h>
#include <MediaZ/Helpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
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
MZ_REGISTER_NAME(PeerID)
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
	std::shared_ptr<CustomVideoSource> mzVideoSource;

	mzWebRTCInterface() {
		webrtcTaskQueue = std::make_shared< AtomicQueue< std::pair<EWebRTCTasks, std::shared_ptr<void>> >>();
		mzVideoSource = std::make_shared<CustomVideoSource>();
		manager = rtc::scoped_refptr<WebRTCManager>(new rtc::RefCountedObject<WebRTCManager>(&client, mzVideoSource.get(), webrtcTaskQueue));
	}
	~mzWebRTCInterface() {
		if(RTCThread.joinable())
			RTCThread.join();
	}
	/// <summary>
	/// This will automatically pus eLOGIN task, do not push it again to queue before/after
	/// </summary>
	void StartConnection(std::string server_port) {
		webrtcTaskQueue->push({ EWebRTCTasks::eLOGIN,std::make_shared<std::string>(server_port)});
		RTCThread = std::thread([this]() {StartRTCThread(); });
	}

	void PushTask(EWebRTCTasks task, std::shared_ptr<void> data) {
		webrtcTaskQueue->push({task,data});
	}
	
	rtc::scoped_refptr<WebRTCManager> manager;
	PeerConnectionClient client;

private:
	std::shared_ptr<AtomicQueue< std::pair<EWebRTCTasks, std::shared_ptr<void>> >> webrtcTaskQueue;
	std::thread RTCThread;
	void StartRTCThread() {
		rtc::WinsockInitializer winsock_init;
		rtc::Win32SocketServer w32_ss;
		rtc::Win32Thread w32_thread(&w32_ss);
		rtc::ThreadManager::Instance()->SetCurrentThread(&w32_thread);

		rtc::InitializeSSL();

		while (manager.get()->MainLoop()) {
			w32_thread.ProcessMessages(1);
		}

		rtc::CleanupSSL();
	}
};

mzWebRTCInterface mzWebRTC;

struct WebRTCNodeContext : mz::NodeContext {
	mzUUID InputPinUUID;
	 mzUUID NodeID;
	std::atomic<bool> shouldSendFrame = false;
	std::mutex Mutex;
	std::thread FrameSenderThread;
	mzResourceShareInfo InputRGBA8 = {};
	mzResourceShareInfo DummyInput = {};
	mzResourceShareInfo Buf = {};
	size_t size;
	uint8_t* data;
	//On Node Created
	WebRTCNodeContext(mz::fb::Node const* node) :NodeContext(node){
		InputRGBA8.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		InputRGBA8.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		InputRGBA8.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
		InputRGBA8.Info.Texture.Width = 640;
		InputRGBA8.Info.Texture.Height = 360;

		mzEngine.Create(&InputRGBA8);

		Buf.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		size = InputRGBA8.Info.Texture.Width * InputRGBA8.Info.Texture.Height * 4 * sizeof(uint8_t);
		Buf.Info.Buffer.Size = size;
		Buf.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		data = new uint8_t[size];
		mzEngine.Create(&Buf);

		DummyInput.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		DummyInput.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		for (auto pin : *node->pins()) {
			if(pin->show_as() == mz::fb::ShowAs::INPUT_PIN){
				InputPinUUID = *pin->id();
			}
		}
		NodeID = *node->id();
		mzWebRTC.manager->SetPeerConnectedCallback(std::bind(&WebRTCNodeContext::OnPeerConnected, this));
		mzWebRTC.manager->SetPeerDisconnectedCallback(std::bind(&WebRTCNodeContext::OnPeerDisconnected, this));
		
	}

	~WebRTCNodeContext() override {

		delete[] data;
	}

	void  OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {
		
		if (pinName == MZN_In) {
			DummyInput = mz::DeserializeTextureInfo(value->Data);
		}
	}

	void OnPinConnected(mz::Name pinName, mzUUID connectedPin) override
	{
		
	}


	mzResult BeginCopyTo(mzCopyInfo* cpy) override {
		cpy->ShouldCopyTexture = true;
		cpy->CopyTextureFrom = DummyInput;
		cpy->CopyTextureTo = InputRGBA8;
		cpy->ShouldSubmitAndWait = true;
		return MZ_RESULT_SUCCESS;
	}

	void OnPeerConnected() {
		mzEngine.LogI("WebRTC client starts frame thread");
		if (!FrameSenderThread.joinable()) {
			shouldSendFrame = true;
			FrameSenderThread = std::thread([this]() {SendFrames(); });
		}
	}

	void OnPeerDisconnected() {
		shouldSendFrame = false;
		FrameSenderThread.join();
	}



	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 2;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC("ConnectToServer");
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
				auto values = mz::GetPinValues(nodeArgs);
				std::string server = mz::GetPinValue<const char>(values, MZN_ServerIP);
				int port = *mz::GetPinValue<int>(values, MZN_Port);
				std::string server_port = server + std::to_string(port);
				mzWebRTC.StartConnection(server_port);
			};

		names[1] = MZ_NAME_STATIC("ConnectToPeer");
		fns[1] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			auto values = mz::GetPinValues(nodeArgs);
			int port = *mz::GetPinValue<int>(values, MZN_PeerID);
			mzWebRTC.PushTask(EWebRTCTasks::eCONNECT, std::make_shared<int>(port));
			};

		return MZ_RESULT_SUCCESS;
	}

	void SendFrames()
	{
		while (shouldSendFrame)
		{
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<mz::app::AppEvent>> Offsets;
			{
				std::unique_lock lock(Mutex);
				mzCmd cmd;
				mzEngine.Begin(&cmd);
				mzEngine.Copy(cmd, &InputRGBA8, &Buf, 0);
				mzEngine.End(cmd);

				auto buf2write = mzEngine.Map(&Buf);
				memcpy(data, buf2write, size);

				//TODO: Write a compute shader for RGBA to YUV420 conversion
				if (data) {
					mzWebRTC.mzVideoSource->PushFrame(buf2write, InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height);
				}

				//std::vector<flatbuffers::Offset<mz::PartialPinUpdate>> updates;
				//updates.push_back(mz::CreatePartialPinUpdateDirect(fbb, &InputPinUUID, 0, false, mz::Action::SET, mz::Action::NOP, 0, 0));

				//mzEngine.HandleEvent(
				//	mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &NodeID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &updates)));

				Offsets.push_back(mz::CreateAppEventOffset(
					fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &InputPinUUID, false)));
				mzEvent hungerEvent = mz::CreateAppEvent(fbb, mz::app::CreateBatchAppEventDirect(fbb, &Offsets));
				mzEngine.EnqueueEvent(&hungerEvent);
			}
		}
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

