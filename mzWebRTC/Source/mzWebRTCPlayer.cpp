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
#include "YUV420toRGB.comp.spv.dat"
#include "mzLinearI420Buffer.h"
#include "mzI420Buffer.h"
#include "mzWebRTCCommon.h"

// mzNodes

MZ_REGISTER_NAME(Out)
MZ_REGISTER_NAME(ServerIP)
MZ_REGISTER_NAME(StreamerID)
MZ_REGISTER_NAME(WebRTCPlayer);
MZ_REGISTER_NAME(TargetBitrate);

MZ_REGISTER_NAME(YUV420toRGB_Compute_Shader);
MZ_REGISTER_NAME(YUV420toRGB_Compute_Pass);
MZ_REGISTER_NAME(Output);
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
struct mzWebRTCPlayerInterface {
public:
	std::chrono::steady_clock::time_point frameReceived;
	rtc::scoped_refptr<mzWebRTCManager> manager;
	mzWebRTCClient client;
	rtc::scoped_refptr<mzCustomVideoSink> mzVideoSink;
	mzWebRTCPlayerInterface() {
		mzVideoSink = rtc::scoped_refptr<mzCustomVideoSink>( new mzCustomVideoSink());
		manager = rtc::scoped_refptr<mzWebRTCManager>(new mzWebRTCManager(&client));
		manager->AddVideoSink(mzVideoSink);
	}
	~mzWebRTCPlayerInterface() {
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

	void SendOffer(int streamerID) {
		if (manager) {
			manager->SendOffer(streamerID);
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

std::pair<mz::Name, std::vector<uint8_t>> YUV420toRGBShader;

struct WebRTCPlayerNodeContext : mz::NodeContext {
	mzWebRTCStatsLogger playerLogger;

	std::unique_ptr<mzWebRTCPlayerInterface> p_mzWebRTC;
	std::unique_ptr<RingProxy> RGBAConversionRing;
	std::unique_ptr<RingProxy> OutputRing;

	std::atomic<EWebRTCPlayerStates> currentState;
	mzUUID NodeID;
	mzUUID OutputPinUUID;
	mzUUID ConnectToServerID;
	mzUUID ConnectToPeerID;
	mzUUID DisconnectFromServerID;

	std::atomic<bool> shouldConvertFrame = false;
	std::atomic<bool> checkCallbacks = true;

	std::mutex WebRTCCallbacksMutex;
	std::condition_variable WebRTCCallbacksCV;

	std::mutex FrameConversionMutex;
	std::condition_variable FrameConversionCV;

	std::thread FrameConverterThread;
	std::thread CallbackHandlerThread;

	mzResourceShareInfo OutputRGBA8 = {};

	std::vector<mzResourceShareInfo> ConvertedTextures = {};
	std::vector<mzResourceShareInfo> OnFrameYBuffers;
	std::vector<mzResourceShareInfo> OnFrameUBuffers;
	std::vector<mzResourceShareInfo> OnFrameVBuffers;

	std::atomic_int PeerCount = 0;
	std::string server;
	std::mutex IsFrameCopyable;

	//On Node Created
	WebRTCPlayerNodeContext(mz::fb::Node const* node) :NodeContext(node), currentState(EWebRTCPlayerStates::eNONE), playerLogger("WebRTC Player") {
		OutputRGBA8.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		OutputRGBA8.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		OutputRGBA8.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
		OutputRGBA8.Info.Texture.Width = 960;
		OutputRGBA8.Info.Texture.Height = 540;

		mzEngine.Create(&OutputRGBA8);

		for (int i = 0; i < 5; i++) {
			mzResourceShareInfo tex = {};
			tex.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
			tex.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			tex.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
			tex.Info.Texture.Width = 960;
			tex.Info.Texture.Height = 540;
			mzEngine.Create(&tex);
			ConvertedTextures.push_back(std::move(tex));

			mzResourceShareInfo tmpY = {};
			mzResourceShareInfo tmpU = {};
			mzResourceShareInfo tmpV = {};
			OnFrameYBuffers.push_back(std::move(tmpY));
			OnFrameUBuffers.push_back(std::move(tmpU));
			OnFrameVBuffers.push_back(std::move(tmpV));
		}

		RGBAConversionRing = std::make_unique<RingProxy>(OnFrameYBuffers.size());
		OutputRing = std::make_unique<RingProxy>(ConvertedTextures.size());

		for (auto pin : *node->pins()) {
			if (pin->show_as() == mz::fb::ShowAs::OUTPUT_PIN) {
				OutputPinUUID = *pin->id();
			}
		}
		for (auto func : *node->functions()) {
			if (strcmp(func->class_name()->c_str(), "ConnectToServer") == 0) {
				ConnectToServerID = *func->id();
			}
			else if (strcmp(func->class_name()->c_str(), "ConnectToPeer") == 0) {
				ConnectToPeerID = *func->id();
			}
			else if (strcmp(func->class_name()->c_str(), "DisconnectFromServer") == 0) {
				DisconnectFromServerID = *func->id();
			}
		}
		NodeID = *node->id();
		
		checkCallbacks = true;

		mzEngine.SchedulePin(OutputPinUUID, { 0,1 });

		flatbuffers::FlatBufferBuilder fbb;

		HandleEvent(
			mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &DisconnectFromServerID,
				mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
	}

	~WebRTCPlayerNodeContext() override {
		
		ClearNodeInternals();

		if (CallbackHandlerThread.joinable()) {
			checkCallbacks = false;
			WebRTCCallbacksCV.notify_one();
			CallbackHandlerThread.join();
		}
		for (const auto& text : ConvertedTextures) {
			mzEngine.Destroy(&text);
		}
	}

	void InitializeNodeInternals() {

		p_mzWebRTC.reset(new mzWebRTCPlayerInterface());
		p_mzWebRTC->manager->SetPeerConnectedCallback([this]() {this->OnPeerConnected(); });
		p_mzWebRTC->manager->SetPeerDisconnectedCallback([this]() {this->OnPeerDisconnected(); });
		p_mzWebRTC->manager->SetServerConnectionSuccesfulCallback([this]() {this->OnConnectedToServer(); });
		p_mzWebRTC->manager->SetServerConnectionFailedCallback([this]() {this->OnDisconnectedFromServer(); });
		p_mzWebRTC->mzVideoSink->SetOnFrameCallback([this](const webrtc::VideoFrame& frame) {this->OnVideoFrame(frame); });

		if (!CallbackHandlerThread.joinable()) {
			CallbackHandlerThread = std::thread([this]() {this->HandleWebRTCCallbacks(); });
		}
	}

	void ClearNodeInternals() {
		if (FrameConverterThread.joinable()) {
			shouldConvertFrame = false;
			FrameConverterThread.join();
		}

		p_mzWebRTC.reset();

	}

	void OnVideoFrame(const webrtc::VideoFrame& frame) {
		

		auto buffer = frame.video_frame_buffer()->GetI420();
		if (!RGBAConversionRing->IsWriteable()) {
			mzEngine.LogW("WebRTC Player dropped a frame");
			return;
		}

		int writeIndex = RGBAConversionRing->GetNextWritable();

		if (OnFrameYBuffers[writeIndex].Info.Texture.Width != buffer->width() || OnFrameYBuffers[writeIndex].Info.Texture.Height != buffer->height()) {
			if(OnFrameYBuffers[writeIndex].Memory.Handle != NULL) {
				mzEngine.Destroy(&OnFrameYBuffers[writeIndex]);
				mzEngine.Destroy(&OnFrameUBuffers[writeIndex]);
				mzEngine.Destroy(&OnFrameVBuffers[writeIndex]);
			}
			OnFrameYBuffers[writeIndex].Info.Texture.Format = MZ_FORMAT_R8_SRGB;
			OnFrameYBuffers[writeIndex].Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			OnFrameYBuffers[writeIndex].Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
			OnFrameYBuffers[writeIndex].Info.Texture.Width = buffer->width();
			OnFrameYBuffers[writeIndex].Info.Texture.Height = buffer->height();
			mzEngine.Create(&OnFrameYBuffers[writeIndex]);

			OnFrameUBuffers[writeIndex].Info.Texture.Format = MZ_FORMAT_R8_SRGB;
			OnFrameUBuffers[writeIndex].Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			OnFrameUBuffers[writeIndex].Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
			OnFrameUBuffers[writeIndex].Info.Texture.Width = buffer->width() / 2;
			OnFrameUBuffers[writeIndex].Info.Texture.Height = buffer->height() / 2;
			mzEngine.Create(&OnFrameUBuffers[writeIndex]);

			OnFrameVBuffers[writeIndex].Info.Texture.Format = MZ_FORMAT_R8_SRGB;
			OnFrameVBuffers[writeIndex].Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
			OnFrameVBuffers[writeIndex].Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
			OnFrameVBuffers[writeIndex].Info.Texture.Width = buffer->width() / 2;
			OnFrameVBuffers[writeIndex].Info.Texture.Height = buffer->height() / 2;
			mzEngine.Create(&OnFrameVBuffers[writeIndex]);
		}
		

		mzEngine.ImageLoad(buffer->DataY(), mzVec2u(buffer->width(), buffer->height()), MZ_FORMAT_R8_SRGB, &OnFrameYBuffers[writeIndex]);
		mzEngine.ImageLoad(buffer->DataU(), mzVec2u(buffer->width() / 2, buffer->height() / 2), MZ_FORMAT_R8_SRGB, &OnFrameUBuffers[writeIndex]);
		mzEngine.ImageLoad(buffer->DataV(), mzVec2u(buffer->width() / 2, buffer->height() / 2), MZ_FORMAT_R8_SRGB, &OnFrameVBuffers[writeIndex]);
		RGBAConversionRing->SetWrote();
		FrameConversionCV.notify_one();
	}

	void  OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {

		
	}

	mzResult BeginCopyFrom(mzCopyInfo* copyInfo) override{
		if (!OutputRing->IsReadable()) {
			return MZ_RESULT_FAILED;
		}

		playerLogger.LogStats();
		int readIndex = OutputRing->GetNextReadable();
		copyInfo->ShouldCopyTexture = true;
		copyInfo->CopyTextureFrom = ConvertedTextures[readIndex];
		copyInfo->CopyTextureTo = mz::DeserializeTextureInfo(copyInfo->SrcPinData.Data);
		copyInfo->ShouldSubmitAndWait = true;
		return MZ_RESULT_SUCCESS;
	}

	void EndCopyFrom(mzCopyInfo* cpy) override {
		OutputRing->SetRead();
		FrameConversionCV.notify_one();
	}

	void ConvertFrames() {
		while (shouldConvertFrame) {
			//mzEngine.WatchLog("WebRTC Player Conversion Ring Readable Size: ", std::to_string(RGBAConversionRing->Size - RGBAConversionRing->FreeCount).c_str());
			//mzEngine.WatchLog("WebRTC Player Conversion Ring Writable Size: ", std::to_string(RGBAConversionRing->FreeCount).c_str());
			//mzEngine.WatchLog("WebRTC Player Output Ring Readable Size: ", std::to_string(OutputRing->Size - OutputRing->FreeCount).c_str());
			//mzEngine.WatchLog("WebRTC Player Output Ring Writable Size: ", std::to_string(OutputRing->FreeCount).c_str());
			if (!RGBAConversionRing->IsReadable() || !OutputRing->IsWriteable()) {
				std::unique_lock<std::mutex> lock(FrameConversionMutex);
				FrameConversionCV.wait(lock);
				continue;
			}

			int readIndex = RGBAConversionRing->GetNextReadable();
			auto t_start = std::chrono::high_resolution_clock::now();
			int writeIndex = OutputRing->GetNextWritable();
			std::vector<mzShaderBinding> inputs;
			inputs.emplace_back(mz::ShaderBinding(MZN_Output, ConvertedTextures[writeIndex]));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneY, OnFrameYBuffers[readIndex]));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneU, OnFrameUBuffers[readIndex]));
			inputs.emplace_back(mz::ShaderBinding(MZN_PlaneV, OnFrameVBuffers[readIndex]));


			mzCmd cmdRunPass;
			mzEngine.Begin(&cmdRunPass);
			auto t0 = std::chrono::high_resolution_clock::now();
			{
				mzRunComputePassParams pass = {};
				pass.Key = MZN_YUV420toRGB_Compute_Pass;
				pass.DispatchSize = mzVec2u(OutputRGBA8.Info.Texture.Width / 24, OutputRGBA8.Info.Texture.Height / 12);
				pass.Bindings = inputs.data();
				pass.BindingCount = inputs.size();
				pass.Benchmark = 0;
				mzEngine.RunComputePass(cmdRunPass, &pass);
			}
			mzEngine.End(cmdRunPass);
			OutputRing->SetWrote();

			//auto t1 = std::chrono::high_resolution_clock::now();
			//mzEngine.WatchLog("WebRTC Player-Compute Pass Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());
			//auto t_end = std::chrono::high_resolution_clock::now();
			//
			//mzEngine.WatchLog("WebRTC Player Run Time(us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()).c_str());
			RGBAConversionRing->SetRead();
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
		*count = 3;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		names[0] = MZ_NAME_STATIC("ConnectToServer");
		fns[0] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
				if (WebRTCPlayerNodeContext* playerNode = static_cast<WebRTCPlayerNodeContext*>(ctx)) {
					auto values = mz::GetPinValues(nodeArgs);
					
					playerNode->InitializeNodeInternals();
					playerNode->server = mz::GetPinValue<const char>(values, MZN_ServerIP);
					playerNode->p_mzWebRTC->StartConnection(playerNode->server);
				}
				
			};

		names[1] = MZ_NAME_STATIC("DisconnectFromServer");
		fns[1] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			if (WebRTCPlayerNodeContext* playerNode = static_cast<WebRTCPlayerNodeContext*>(ctx)) {
				auto values = mz::GetPinValues(nodeArgs);

				playerNode->currentState = EWebRTCPlayerStates::eDISCONNECTED_FROM_SERVER;
				playerNode->WebRTCCallbacksCV.notify_one();
			}

			};

		names[2] = MZ_NAME_STATIC("ConnectToPeer");
		fns[2] = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			if (WebRTCPlayerNodeContext* playerNode = static_cast<WebRTCPlayerNodeContext*>(ctx)) {
				auto values = mz::GetPinValues(nodeArgs);

				playerNode->p_mzWebRTC->SendOffer(*mz::GetPinValue<int>(values, MZN_StreamerID));
			}

			};

		return MZ_RESULT_SUCCESS;
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
					mzEngine.LogI("WebRTC Player connected to server");

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
					if (!FrameConverterThread.joinable()) {
						shouldConvertFrame = true;
						FrameConverterThread = std::thread([this]() {ConvertFrames(); });
					}
					flatbuffers::FlatBufferBuilder fbb;
					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToPeerID,
							mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, true))));
					currentState = EWebRTCPlayerStates::eNONE;
					break;
				}
				case EWebRTCPlayerStates::eDISCONNECTED_FROM_SERVER:
				{
					mzEngine.LogI("WebRTC Player disconnected from server");

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
					flatbuffers::FlatBufferBuilder fbb;
					HandleEvent(
						mz::CreateAppEvent(fbb, mz::CreatePartialNodeUpdateDirect(fbb, &ConnectToPeerID,
							mz::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0, mz::fb::CreateOrphanStateDirect(fbb, false))));

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
		.Key = YUV420toRGBShader.first,
		.Source = {.SpirvBlob = { YUV420toRGBShader.second.data(), YUV420toRGBShader.second.size() }},
		};
		return MZ_RESULT_SUCCESS;

	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes) {

		*count = 1;

		if (!passes)
			return MZ_RESULT_SUCCESS;
		*passes++ = {
			.Key = MZN_YUV420toRGB_Compute_Pass, .Shader = MZN_YUV420toRGB_Compute_Shader, .MultiSample = 1
		};

		return MZ_RESULT_SUCCESS;
	};
};

void RegisterWebRTCPlayer(mzNodeFunctions* outFunctions) {
	MZ_BIND_NODE_CLASS(MZN_WebRTCPlayer, WebRTCPlayerNodeContext, outFunctions);

	YUV420toRGBShader = { MZN_YUV420toRGB_Compute_Shader, {std::begin(YUV420toRGB_comp_spv), std::end(YUV420toRGB_comp_spv)} };
}
