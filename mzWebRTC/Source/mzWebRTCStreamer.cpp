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

#include "rtc_base/checks.h"
#include "rtc_base/ssl_adapter.h"
#include "rtc_base/string_utils.h"  // For ToUtf8
#include "rtc_base/win32_socket_init.h"
#include "system_wrappers/include/field_trial.h"
#include "test/field_trial.h"
#include "mzCustomVideoSource.h"
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
#include "mzWebRTCManager.h"
#include "mzWebRTCClient.h"
#include "RGBtoYUV420_Linearized.comp.spv.dat"
#include "mzLinearI420Buffer.h"
#include "mzI420Buffer.h"
#include "mzWebRTCCommon.h"

// mzNodes

MZ_REGISTER_NAME(In)
MZ_REGISTER_NAME(ServerIP)
MZ_REGISTER_NAME(MaxFPS)
MZ_REGISTER_NAME(WebRTCStreamer);
MZ_REGISTER_NAME(TargetBitrate);

MZ_REGISTER_NAME(RGBtoYUV420_Compute_Shader);
MZ_REGISTER_NAME(RGBtoYUV420_Compute_Pass);
MZ_REGISTER_NAME(Input);
MZ_REGISTER_NAME(PlaneY);
MZ_REGISTER_NAME(PlaneU);
MZ_REGISTER_NAME(PlaneV);

enum EWebRTCPlayerStates {
	eNONE,
	eREQUESTED_TO_CONNECT_SERVER,
	eCONNECTED_TO_SERVER,
	eCONNECTED_TO_PEER,
	eDISCONNECTED_FROM_SERVER,
	eDISCONNECTED_FROM_PEER,
};

//The interface between medaiZ and WebRTC, stores the task qeueue and launches the connection thread
struct mzWebRTCStreamerInterface {
public:
	rtc::scoped_refptr<mzWebRTCManager> manager;
	mzWebRTCClient client;
	rtc::scoped_refptr<mzCustomVideoSource> mzVideoSource;

	mzWebRTCStreamerInterface() {
		mzVideoSource = rtc::scoped_refptr<mzCustomVideoSource>( new mzCustomVideoSource());
		manager = rtc::scoped_refptr<mzWebRTCManager>(new mzWebRTCManager(&client));
		manager->AddVideoSource(mzVideoSource);
	}
	~mzWebRTCStreamerInterface() {
		manager->Dispose();
		Dispose();
	}
	
	void StartConnection(std::string server_port) {

		if(!RTCThread.joinable())
			RTCThread = std::thread([this]() {this->StartRTCThread(); });
		try {
			client.ConnectToServer(server_port);
		}
		catch (std::exception& E) {
			mzEngine.LogE(E.what());
		}
	}

	void SetTargetBitrate(int kbps) {
		if (manager) {
			manager->UpdateBitrates(kbps);
		}
	}

	void Dispose() {
		isAlive = false;
		if (RTCThread.joinable())
			RTCThread.join();
	}
	

private:
	std::atomic<bool> isAlive = true;
	std::thread RTCThread;
	void StartRTCThread() {
		rtc::WinsockInitializer winsock_init;
		rtc::Win32SocketServer w32_ss;
		rtc::Win32Thread w32_thread(&w32_ss);
		rtc::ThreadManager::Instance()->SetCurrentThread(&w32_thread);

		while (isAlive) {
			client.Update();
			manager->MainLoop();
			w32_thread.ProcessMessages(1);
		}
		
		w32_thread.Quit();
		rtc::ThreadManager::Instance()->SetCurrentThread(nullptr);
		
		//OpenSSL v1.1.1x should clean itself from memory but
		//just for safety we will call this
	}
};

//TODO: mvoe this to node context to allow multiple streamers!!!
std::pair<mz::Name, std::vector<uint8_t>> RGBtoYUV420Shader;

struct WebRTCNodeContext : mz::NodeContext {
	
	mzWebRTCStatsLogger encodeLogger;
	mzWebRTCStatsLogger copyToLogger;

	std::unique_ptr<mzWebRTCStreamerInterface> p_mzWebRTC;
	std::unique_ptr<RingProxy> InputRing;
	std::chrono::microseconds interFrameTimeDelta;
	std::chrono::microseconds timeLimit;
	std::chrono::steady_clock::time_point encodeStartTime;

	size_t nextBufferToCopyIndex;
	std::vector<rtc::scoped_refptr<mzI420Buffer>> buffers;
	std::atomic<size_t> FreeBuffers;

	std::mutex EncodeMutex;
	std::condition_variable EncodeCompletedCV;

	std::atomic_bool CopyCompleted = false;

	std::atomic<EWebRTCPlayerStates> currentState;
	mzUUID InputPinUUID;
	mzUUID NodeID;
	mzUUID ConnectToServerID;
	mzUUID DisconnectFromServerID;

	std::atomic<bool> shouldSendFrame = false;
	std::atomic<bool> shouldSendHunger = true;
	std::atomic<bool> checkCallbacks = true;

	std::mutex WebRTCCallbacksMutex;
	std::condition_variable WebRTCCallbacksCV;

	std::mutex SendFrameMutex;
	std::condition_variable SendFrameCV;

	std::thread FrameSenderThread;
	std::thread CallbackHandlerThread;

	std::mutex RingNewFrameMutex;
	std::condition_variable RingNewFrameCV;

	mzResourceShareInfo InputRGBA8 = {};
	mzResourceShareInfo DummyInput = {}; 
	
	std::vector<mzResourceShareInfo> InputBuffers = {};
	std::vector<mzResourceShareInfo> YUVPlanes = {};
	std::vector<mzResourceShareInfo> YUVBuffers = {};

	float FPS;
	std::atomic_int PeerCount = 0;
	std::string server;
	std::atomic_bool StopRequested = false;
	//On Node Created
	WebRTCNodeContext(mz::fb::Node const* node) :NodeContext(node), currentState(EWebRTCPlayerStates::eNONE), encodeLogger("WebRTC Streamer Encode"), copyToLogger("WebRTC Stramer BeginCopyTo: ") {
		InputRGBA8.Info.Texture.Format = MZ_FORMAT_B8G8R8A8_SRGB;
		InputRGBA8.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		InputRGBA8.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
		InputRGBA8.Info.Texture.Width = 1280;
		InputRGBA8.Info.Texture.Height = 720;

		mzEngine.Create(&InputRGBA8);

		DummyInput.Info.Texture.Format = MZ_FORMAT_B8G8R8A8_SRGB;
		DummyInput.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;

		buffers.push_back(new mzI420Buffer(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height));
		buffers.push_back(new mzI420Buffer(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height));
		buffers.push_back(new mzI420Buffer(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height));
		buffers.push_back(new mzI420Buffer(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height));
		buffers.push_back(new mzI420Buffer(InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height));

		for (int i = 0; i < buffers.size(); i++) {
			mzResourceShareInfo PlaneY = {};
			PlaneY.Info.Texture.Format = MZ_FORMAT_R8_SRGB;
			PlaneY.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			PlaneY.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
			PlaneY.Info.Texture.Width = InputRGBA8.Info.Texture.Width;
			PlaneY.Info.Texture.Height = InputRGBA8.Info.Texture.Height + InputRGBA8.Info.Texture.Height / 2;
			mzEngine.Create(&PlaneY);
			
			mzResourceShareInfo BufY  = {};
			BufY.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
			BufY.Info.Buffer.Size = PlaneY.Info.Texture.Width * PlaneY.Info.Texture.Height * sizeof(uint8_t);
			BufY.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
			mzEngine.Create(&BufY);

			mzResourceShareInfo Input = {};
			Input.Info.Texture.Format = MZ_FORMAT_B8G8R8A8_SRGB;
			Input.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			Input.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
			Input.Info.Texture.Width = InputRGBA8.Info.Texture.Width;
			Input.Info.Texture.Height = InputRGBA8.Info.Texture.Height;
			mzEngine.Create(&Input);

			YUVBuffers.push_back(std::move(BufY));
			YUVPlanes.push_back(std::move(PlaneY));
			InputBuffers.push_back(std::move(Input));
		}

		InputRing = std::make_unique<RingProxy>(InputBuffers.size());
		InputRing->SetConditionVariable(&RingNewFrameCV);

		FreeBuffers = buffers.size();
		nextBufferToCopyIndex = 0;

		for (auto pin : *node->pins()) {
			if (pin->show_as() == mz::fb::ShowAs::INPUT_PIN) {
				InputPinUUID = *pin->id();
			}
			if (MZN_MaxFPS.Compare(pin->name()->c_str()) == 0)
			{
				FPS = *(float*)pin->data()->data();
				auto time = std::chrono::duration<float>(1.0f / FPS);
				timeLimit = std::chrono::round<std::chrono::microseconds>(time);
			}
		}
		for (auto func : *node->functions()) {
			if (strcmp(func->class_name()->c_str(), "ConnectToServer") == 0) {
				ConnectToServerID = *func->id();
			}
			else if (strcmp(func->class_name()->c_str(), "DisconnectFromServer") == 0) {
				DisconnectFromServerID = *func->id();
			}
		}
		NodeID = *node->id();
		
		checkCallbacks = true;
		
		mzVec2u deltaSec{10'000u, (uint32_t)std::floor(FPS * 10'000)};
		mzEngine.SchedulePin(InputPinUUID, deltaSec);

		flatbuffers::FlatBufferBuilder fbb;

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &DisconnectFromServerID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
	}

	~WebRTCNodeContext() override {
		
		ClearNodeInternals();

		if (CallbackHandlerThread.joinable()) {
			checkCallbacks = false;
			WebRTCCallbacksCV.notify_one();
			CallbackHandlerThread.join();
		}
		for (auto& yuvBuf : YUVBuffers) {
			mzEngine.Destroy(&yuvBuf);
		}
		for (auto& yuvPlane : YUVPlanes) {
			mzEngine.Destroy(&yuvPlane);
		}
	}

	void InitializeNodeInternals() {

		p_mzWebRTC.reset(new mzWebRTCStreamerInterface());
		p_mzWebRTC->manager->SetPeerConnectedCallback([this]() {this->OnPeerConnected(); });
		p_mzWebRTC->manager->SetPeerDisconnectedCallback([this]() {this->OnPeerDisconnected(); });
		p_mzWebRTC->manager->SetServerConnectionSuccesfulCallback([this]() {this->OnConnectedToServer(); });
		p_mzWebRTC->manager->SetServerConnectionFailedCallback([this]() {this->OnDisconnectedFromServer(); });
		p_mzWebRTC->manager->SetImageEncodeCompletedCallback([this]() {this->OnEncodeCompleted(); });

		if (!CallbackHandlerThread.joinable()) {
			CallbackHandlerThread = std::thread([this]() {this->HandleWebRTCCallbacks(); });
		}
	}

	void ClearNodeInternals() {
		if (FrameSenderThread.joinable()) {
			SendFrameCV.notify_one();
			shouldSendFrame = false;
			FrameSenderThread.join();
		}

		p_mzWebRTC.reset();

	}

	void  OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {

		if (pinName == MZN_In) {
			DummyInput = mz::DeserializeTextureInfo(value->Data);
		}
		if (pinName == MZN_MaxFPS) {
			FPS = *(static_cast<float*>(value->Data));
			auto time = std::chrono::duration<float>(1.0f / FPS);
			timeLimit = std::chrono::round<std::chrono::microseconds>(time);
		}
		if (pinName == MZN_TargetBitrate) {
			int targetKbps = *(static_cast<int*>(value->Data));
			p_mzWebRTC->SetTargetBitrate(targetKbps);
		}
	}

	void OnPinConnected(mz::Name pinName, mzUUID connectedPin) override
	{

	}

	void OnConnectedToServer() {
		currentState = EWebRTCPlayerStates::eCONNECTED_TO_SERVER;
		WebRTCCallbacksCV.notify_one();
	}

	void OnDisconnectedFromServer() {
		currentState = EWebRTCPlayerStates::eDISCONNECTED_FROM_SERVER;
		WebRTCCallbacksCV.notify_one();
		//p_mzWebRTC.reset(new mzWebRTCInterface());
	}

	void OnPeerConnected() {
		++PeerCount;
		currentState = EWebRTCPlayerStates::eCONNECTED_TO_PEER;
		WebRTCCallbacksCV.notify_one();
	}

	void OnPeerDisconnected() {
		--PeerCount;
		currentState = EWebRTCPlayerStates::eDISCONNECTED_FROM_PEER;
		WebRTCCallbacksCV.notify_one();
	}



	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 2;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC("ConnectToServer");
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
				if (WebRTCNodeContext* streamerNode = static_cast<WebRTCNodeContext*>(ctx)) {
					auto values = mz::GetPinValues(nodeArgs);
					
					streamerNode->InitializeNodeInternals();
					streamerNode->server = mz::GetPinValue<const char>(values, MZN_ServerIP);
					streamerNode->p_mzWebRTC->StartConnection(streamerNode->server);
				}
				
			};

		names[1] = MZ_NAME_STATIC("DisconnectFromServer");
		fns[1] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			if (WebRTCNodeContext* streamerNode = static_cast<WebRTCNodeContext*>(ctx)) {
				auto values = mz::GetPinValues(nodeArgs);

				streamerNode->currentState = EWebRTCPlayerStates::eDISCONNECTED_FROM_SERVER;
				streamerNode->WebRTCCallbacksCV.notify_one();
			}

			};

		return MZ_RESULT_SUCCESS;
	}

	mzResult BeginCopyTo(mzCopyInfo* cpy) override {
		copyToLogger.LogStats();
		if (!InputRing->IsWriteable()) {
			mzEngine.LogW("WebRTC Streamer frame drop!");
			return MZ_RESULT_FAILED;
		}

		int writeIndex = InputRing->GetNextWritable();
		cpy->ShouldCopyTexture = true;
		cpy->CopyTextureFrom = DummyInput;
		cpy->CopyTextureTo = InputBuffers[writeIndex];
		cpy->ShouldSubmitAndWait = true;
		cpy->Stop = false;
		return MZ_RESULT_SUCCESS;
	}

	void EndCopyTo(mzCopyInfo* cpy) override {
		if (!InputRing->IsWriteable()) {
		}
		CopyCompleted = true;
		InputRing->SetWrote();
		SendFrameCV.notify_one();
		StopRequested = !InputRing->IsWriteable();
	}

	void OnEncodeCompleted() {
		
		encodeLogger.LogStats();

		FreeBuffers++;
		EncodeCompletedCV.notify_one();
	}

	void SendFrames()
	{
		std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();
		std::chrono::microseconds passedTime;
		while (shouldSendFrame && p_mzWebRTC)
		{
			while (true)
			{
				InputRing->LogRing();
				passedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime);
				if(passedTime.count() < timeLimit.count())
					continue;
				break;
			}
			startTime = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("WebRTC Streamer interframe passed time:", std::to_string(passedTime.count()).c_str());
			if (PeerCount == 0)
				continue;

			auto t_start = std::chrono::high_resolution_clock::now();
			
			if(!InputRing->IsReadable())
			{
				mzEngine.LogW("WebRTC Streamer dropped a frame");
				continue;
			}

			int readIndex = InputRing->GetNextReadable();

			std::vector<mzShaderBinding> inputs;
			inputs.emplace_back(mz::ShaderBinding(MZN_Input, InputBuffers[readIndex]));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneY, YUVPlanes[nextBufferToCopyIndex]));

			mzCmd cmdRunPass; 
			mzEngine.Begin(&cmdRunPass);
			auto t0 = std::chrono::high_resolution_clock::now();

			{
				mzRunComputePassParams pass = {};
				pass.Key = MZN_RGBtoYUV420_Compute_Pass;
				pass.DispatchSize = mzVec2u(InputRGBA8.Info.Texture.Width/24, InputRGBA8.Info.Texture.Height/12);
				pass.Bindings = inputs.data();
				pass.BindingCount = inputs.size();
				pass.Benchmark = 0;
				mzEngine.RunComputePass(cmdRunPass, &pass);
			}
			auto t1 = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("WebRTC Streamer-Compute Pass Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			t0 = std::chrono::high_resolution_clock::now();
			{
				mzResourceShareInfo tempBuf = {};

				mzEngine.Copy(cmdRunPass, &YUVPlanes[nextBufferToCopyIndex], &YUVBuffers[nextBufferToCopyIndex], 0);

				mzEngine.End(cmdRunPass);

				//mzVec2u deltaSec{ 10'000u, (uint32_t)std::floor(FPS * 10'000) };
				//mzEngine.SchedulePin(InputPinUUID, deltaSec);
				InputRing->SetRead();
				/*if (StopRequested && InputRing->IsWriteable()) {
					mzVec2u deltaSec{ 10'000u, (uint32_t)std::floor(FPS * 10'000) };
					mzEngine.SchedulePin(InputPinUUID, deltaSec);
				}*/


				auto dataY = mzEngine.Map(&YUVBuffers[nextBufferToCopyIndex]);
				bool isY = (dataY != nullptr);

				auto dataU = dataY + InputRGBA8.Info.Texture.Width * InputRGBA8.Info.Texture.Height;
				bool isU = (dataU != nullptr);

				auto dataV = dataU + InputRGBA8.Info.Texture.Width / 2 * InputRGBA8.Info.Texture.Height / 2;
				bool isV = (dataV != nullptr);

				if (!(isY && isU && isV)) {
					mzEngine.LogE("YUV420 Frame can not be built!");
					return;
				}

				auto yuvBuffer = buffers[nextBufferToCopyIndex++];
				//wait for encode
				yuvBuffer->SetDataY(dataY);
				nextBufferToCopyIndex %= buffers.size();
				FreeBuffers--;
				
				t1 = std::chrono::high_resolution_clock::now();
				//mzEngine.WatchLog("WebRTC Streamer-YUV420 Frame Build Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());


				webrtc::VideoFrame frame =
					webrtc::VideoFrame::Builder()
					.set_video_frame_buffer(yuvBuffer)
					.set_rotation(webrtc::kVideoRotation_0)
					.set_timestamp_us(rtc::TimeMicros())
					.build();

				p_mzWebRTC->mzVideoSource->PushFrame(frame);

				[[unlikely]]
				if (FreeBuffers == 0 && PeerCount > 0)
				{
					std::unique_lock lock(EncodeMutex);
					EncodeCompletedCV.wait(lock);
				}
			}

			auto t_end = std::chrono::high_resolution_clock::now();

			//mzEngine.WatchLog("WebRTC Streamer Run Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()).c_str());
			//mzEngine.WatchLog("WebRTC Streamer FPS", std::to_string(1.0/passedTime.count()*1'000'000.0).c_str());
		}
	}

	void HandleWebRTCCallbacks() {
		while (checkCallbacks) {
			
			std::unique_lock<std::mutex> lck(WebRTCCallbacksMutex);
			WebRTCCallbacksCV.wait(lck);
			
			switch (currentState) {
				case EWebRTCPlayerStates::eNONE: 
				{
					//Idle
					break;
				}
				case EWebRTCPlayerStates::eREQUESTED_TO_CONNECT_SERVER:
				{
					p_mzWebRTC->StartConnection(server);
					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}
				case EWebRTCPlayerStates::eCONNECTED_TO_SERVER:
				{
					mzEngine.LogI("WebRTC Streamer connected to server");

					flatbuffers::FlatBufferBuilder fbb;
					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToServerID, 
							mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));

					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &DisconnectFromServerID,
							mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));

					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}
				case EWebRTCPlayerStates::eCONNECTED_TO_PEER: 
				{
					if (!FrameSenderThread.joinable()) {
						mzEngine.LogI("WebRTC Streamer starts frame thread");
						shouldSendFrame = true;
						FrameSenderThread = std::thread([this]() {SendFrames(); });
						flatbuffers::FlatBufferBuilder fbb;
						HandleEvent(mz::CreateAppEvent(fbb, mz::app::CreateSetThreadNameDirect(fbb, (u64)FrameSenderThread.native_handle(), "WebRTC Frame Sender")));
					}
					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}
				case EWebRTCPlayerStates::eDISCONNECTED_FROM_SERVER:
				{
					mzEngine.LogI("WebRTC Streamer disconnected from server");

					flatbuffers::FlatBufferBuilder fbb;
					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToServerID, mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));
					
					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &DisconnectFromServerID,
							mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
					
					ClearNodeInternals();
					
					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}
				case EWebRTCPlayerStates::eDISCONNECTED_FROM_PEER:
				{
					//shouldSendFrame = false;
					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}

			}
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
void RegisterWebRTCStreamer(mzNodeFunctions* outFunctions) {
	MZ_BIND_NODE_CLASS(MZN_WebRTCStreamer, WebRTCNodeContext, outFunctions);

	RGBtoYUV420Shader = { MZN_RGBtoYUV420_Compute_Shader, {std::begin(RGBtoYUV420_Linearized_comp_spv), std::end(RGBtoYUV420_Linearized_comp_spv)} };
}
