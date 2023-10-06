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
#include "api/video/i420_buffer.h"
#include "api/video/video_frame.h"
#include "WebRTCManager.h"
#include "RGBtoYUV420.comp.spv.dat"
// mzNodes

MZ_INIT();
MZ_REGISTER_NAME(In)
MZ_REGISTER_NAME(ServerIP)
MZ_REGISTER_NAME(Port)
MZ_REGISTER_NAME(PeerID)
MZ_REGISTER_NAME(WebRTCClient);

MZ_REGISTER_NAME(RGBtoYUV420_Compute_Shader);
MZ_REGISTER_NAME(RGBtoYUV420_Compute_Pass);
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(PlaneY);
MZ_REGISTER_NAME(PlaneU);
MZ_REGISTER_NAME(PlaneV);

#define ConnectToServer  "ConnectToServer"
#define ConnectToPeer "ConnectToPeer"

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

		if(!RTCThread.joinable())
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
std::pair<mz::Name, std::vector<uint8_t>> RGBtoYUV420Shader;

struct WebRTCNodeContext : mz::NodeContext {
	mzUUID InputPinUUID;
	mzUUID NodeID;
	mzUUID ConnectToServerID;
	mzUUID ConnectToPeerID;
	std::atomic<bool> shouldSendFrame = false;
	std::atomic<bool> shouldSendHunger = true;
	std::mutex Mutex;
	
	std::thread FrameSenderThread;
	std::thread HungerSenderThread;

	mzResourceShareInfo InputRGBA8 = {};
	mzResourceShareInfo DummyInput = {};
	mzResourceShareInfo Buf = {};
	
	mzResourceShareInfo PlaneY = {};
	mzResourceShareInfo PlaneU = {};
	mzResourceShareInfo PlaneV = {};

	mzResourceShareInfo BufY = {};
	mzResourceShareInfo BufU = {};
	mzResourceShareInfo BufV = {};

	size_t size;
	uint8_t* data;
	//On Node Created
	WebRTCNodeContext(mz::fb::Node const* node) :NodeContext(node) {
		InputRGBA8.Info.Texture.Format = MZ_FORMAT_B8G8R8A8_SRGB;
		InputRGBA8.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		InputRGBA8.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
		InputRGBA8.Info.Texture.Width = 960;
		InputRGBA8.Info.Texture.Height = 540;

		mzEngine.Create(&InputRGBA8);

		Buf.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		size = InputRGBA8.Info.Texture.Width * InputRGBA8.Info.Texture.Height * 4 * sizeof(uint8_t);
		Buf.Info.Buffer.Size = size;
		Buf.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);

		data = new uint8_t[size];
		mzEngine.Create(&Buf);

		PlaneY.Info.Texture.Format = MZ_FORMAT_R8_SRGB;
		PlaneY.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		PlaneY.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		PlaneY.Info.Texture.Width = InputRGBA8.Info.Texture.Width;
		PlaneY.Info.Texture.Height = InputRGBA8.Info.Texture.Height;
		mzEngine.Create(&PlaneY);

		PlaneU.Info.Texture.Format = MZ_FORMAT_R8_SRGB;
		PlaneU.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		PlaneU.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		PlaneU.Info.Texture.Width = InputRGBA8.Info.Texture.Width/2;
		PlaneU.Info.Texture.Height = InputRGBA8.Info.Texture.Height/2;
		mzEngine.Create(&PlaneU);

		PlaneV.Info.Texture.Format = MZ_FORMAT_R8_SRGB;
		PlaneV.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		PlaneV.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		PlaneV.Info.Texture.Width = InputRGBA8.Info.Texture.Width/2;
		PlaneV.Info.Texture.Height = InputRGBA8.Info.Texture.Height/2;
		mzEngine.Create(&PlaneV);

		BufY.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		BufY.Info.Buffer.Size = InputRGBA8.Info.Texture.Width * InputRGBA8.Info.Texture.Height * sizeof(uint8_t);
		BufY.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		mzEngine.Create(&BufY);

		BufU.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		BufU.Info.Buffer.Size = InputRGBA8.Info.Texture.Width / 2 * InputRGBA8.Info.Texture.Height / 2 * sizeof(uint8_t);
		BufU.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		mzEngine.Create(&BufU);

		BufV.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		BufV.Info.Buffer.Size = InputRGBA8.Info.Texture.Width / 2 * InputRGBA8.Info.Texture.Height / 2 * sizeof(uint8_t);
		BufV.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		mzEngine.Create(&BufV);

		DummyInput.Info.Texture.Format = MZ_FORMAT_B8G8R8A8_SRGB;
		DummyInput.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;



		for (auto pin : *node->pins()) {
			if (pin->show_as() == mz::fb::ShowAs::INPUT_PIN) {
				InputPinUUID = *pin->id();
			}
		}
		for (auto func : *node->functions()) {
			if (strcmp(func->class_name()->c_str(), ConnectToPeer) == 0) {
				ConnectToPeerID = *func->id();
			}
			else if (strcmp(func->class_name()->c_str(), ConnectToServer) == 0) {
				ConnectToServerID = *func->id();
			}
		}
		NodeID = *node->id();
		mzWebRTC.manager->SetPeerConnectedCallback(std::bind(&WebRTCNodeContext::OnPeerConnected, this));
		mzWebRTC.manager->SetPeerDisconnectedCallback(std::bind(&WebRTCNodeContext::OnPeerDisconnected, this));
		mzWebRTC.manager->SetOnConnectedToServerCallback(std::bind(&WebRTCNodeContext::OnConnectedToServer, this));
		mzWebRTC.manager->SetOnDisconnectedFromServerCallback(std::bind(&WebRTCNodeContext::OnDisconnectedFromServer, this));
		HungerSenderThread = std::thread([this]() {SendHunger(); });
	}

	~WebRTCNodeContext() override {
		if (FrameSenderThread.joinable()) {
			shouldSendFrame = false;
			FrameSenderThread.join();
		}
		if (HungerSenderThread.joinable()) {
			shouldSendHunger = false;
			HungerSenderThread.join();
		}
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

	void EndCopyTo(mzCopyInfo* cpy) override {
		cpy->Stop = true;
	}

	void OnConnectedToServer() {
		mzEngine.LogI("WebRTC Client connected to server");
		flatbuffers::FlatBufferBuilder fbb;
		mzEngine.HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToServerID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
	}

	void OnDisconnectedFromServer() {
		mzEngine.LogI("WebRTC Client disconnected from server");
		flatbuffers::FlatBufferBuilder fbb;
		mzEngine.HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToServerID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));
	}

	void OnPeerConnected() {
		mzEngine.LogI("WebRTC client starts frame thread");
		if (!FrameSenderThread.joinable()) {
			shouldSendFrame = true;
			FrameSenderThread = std::thread([this]() {SendFrames(); });
			flatbuffers::FlatBufferBuilder fbb;
			mzEngine.HandleEvent(
				mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToPeerID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
		}
		else {
			mzEngine.LogW("Multiple peer connection is not allowed at this version!");
		}
	}

	void OnPeerDisconnected() {
		shouldSendFrame = false;
		if (FrameSenderThread.joinable())
			FrameSenderThread.join();


		flatbuffers::FlatBufferBuilder fbb;
		mzEngine.HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToPeerID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));
	}



	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 2;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC(ConnectToServer);
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			auto values = mz::GetPinValues(nodeArgs);
			std::string server = mz::GetPinValue<const char>(values, MZN_ServerIP);
			int port = *mz::GetPinValue<int>(values, MZN_Port);
			std::string server_port = server + std::to_string(port);
			mzWebRTC.StartConnection(server_port);
			};

		names[1] = MZ_NAME_STATIC(ConnectToPeer);
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
			auto t_start = std::chrono::high_resolution_clock::now();
			std::unique_lock lock(Mutex);

			std::vector<mzShaderBinding> inputs;
			inputs.emplace_back(mz::ShaderBinding(MZN_Input, InputRGBA8));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneY, PlaneY));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneU, PlaneU));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneV, PlaneV));

			mzCmd cmdRunPass;
			mzEngine.Begin(&cmdRunPass);

			auto t0 = std::chrono::high_resolution_clock::now();

			{
				mzRunComputePassParams pass = {};
				pass.Key = MZN_RGBtoYUV420_Compute_Pass;
				pass.DispatchSize = mzVec2u(40,45); //local size 16x16, you should make a better way for this
				pass.Bindings = inputs.data();
				pass.BindingCount = inputs.size();
				pass.Benchmark = 0;
				mzEngine.RunComputePass(cmdRunPass, &pass);
			}

			auto t1 = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("WebRTC-Compute Pass Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			mzEngine.End(cmdRunPass);

			rtc::scoped_refptr<webrtc::I420Buffer> buffer =
				webrtc::I420Buffer::Create(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height);
			buffer->InitializeData();

			t0 = std::chrono::high_resolution_clock::now();

			mzCmd cmdBuildFrame;
			mzEngine.Begin(&cmdBuildFrame);
			{
				mzEngine.Copy(cmdBuildFrame, &PlaneY, &BufY, 0);
				auto dataY = mzEngine.Map(&BufY);
				bool isY = (dataY != nullptr);
				int strideY = buffer->StrideY();
				if(isY)
					memcpy(buffer->MutableDataY(), dataY, InputRGBA8.Info.Texture.Width * InputRGBA8.Info.Texture.Height);

				mzEngine.Copy(cmdBuildFrame, &PlaneU, &BufU, 0);
				auto dataU = mzEngine.Map(&BufU);
				bool isU = (dataU != nullptr);
				int strideU = buffer->StrideU();
				if(isU)
					memcpy(buffer->MutableDataU(), dataU, InputRGBA8.Info.Texture.Width/2 * InputRGBA8.Info.Texture.Height/2);

				mzEngine.Copy(cmdBuildFrame, &PlaneV, &BufV, 0);
				auto dataV = mzEngine.Map(&BufV);
				bool isV = (dataV != nullptr);
				int strideV = buffer->StrideV();
				if(isV)
					memcpy(buffer->MutableDataV(), dataV, InputRGBA8.Info.Texture.Width/2 * InputRGBA8.Info.Texture.Height/2);
				
				if (!(isY && isU && isV)) {
					mzEngine.LogE("YUV420 Frame can not be built!");
					continue;
				}

			}

			t1 = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("WebRTC-YUV420 Frame Build Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			mzEngine.End(cmdBuildFrame);


			webrtc::VideoFrame frame =
				webrtc::VideoFrame::Builder()
				.set_video_frame_buffer(buffer)
				.set_rotation(webrtc::kVideoRotation_0)
				.set_timestamp_us(rtc::TimeMicros())
				.build();

			mzWebRTC.mzVideoSource->PushFrame(frame);

			auto t_end = std::chrono::high_resolution_clock::now();

			mzEngine.WatchLog("WebRTC Client Run Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()).c_str());

			

		}
	}

	void SendHunger() {

		while (shouldSendHunger) {
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<mz::app::AppEvent>> Offsets;
			Offsets.push_back(mz::CreateAppEventOffset(
				fbb, mz::app::CreateScheduleRequest(fbb, mz::app::ScheduleRequestKind::PIN, &InputPinUUID, false)));
			mzEvent hungerEvent = mz::CreateAppEvent(fbb, mz::app::CreateBatchAppEventDirect(fbb, &Offsets));
			mzEngine.EnqueueEvent(&hungerEvent);
		}
	}

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders) {
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;

		*outShaders++ = {
		.Key = RGBtoYUV420Shader.first,
		.Source = {.SpirvBlob = { RGBtoYUV420Shader.second.data(), RGBtoYUV420Shader.second.size() }},
		};
		return MZ_RESULT_SUCCESS;

	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes) {

		*count = 1;

		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes++ = {
			.Key = MZN_RGBtoYUV420_Compute_Pass, .Shader = MZN_RGBtoYUV420_Compute_Shader, .MultiSample = 1
		};

		return MZ_RESULT_SUCCESS;
	};
};

extern "C"
{

	MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, mzNodeFunctions** outFunctions) {
		*outCount = (size_t)(1);
		if (!outFunctions)
			return MZ_RESULT_SUCCESS;

		MZ_BIND_NODE_CLASS(MZN_WebRTCClient, WebRTCNodeContext, outFunctions[0]);

		RGBtoYUV420Shader = { MZN_RGBtoYUV420_Compute_Shader, {std::begin(RGBtoYUV420_comp_spv), std::end(RGBtoYUV420_comp_spv)} };

		return MZ_RESULT_SUCCESS;
	}
}
