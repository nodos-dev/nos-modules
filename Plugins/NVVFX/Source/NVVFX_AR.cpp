#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <nosUtil/Thread.h>
#include <nosVulkanSubsystem/Helpers.hpp>
#include "CUDAVulkanInterop.h"
#include "NVVFXAppRunner.h"
#include "NVVFX_Names.h"


// nosNodes


inline void SearchPath(const char* path)
{
#if _WIN32
	::SetDllDirectory(path);
#else
#pragma error "Not implemented"
#endif
}

class StatsLogger
{
public:
	StatsLogger(std::string name, int refreshRate = 100) : RefreshRate(refreshRate)
	{
		Name_FPS = name + " FPS: ";
		Name_MAX_FPS = name + " MAX FPS: ";
		Name_MIN_FPS = name + " MIN FPS: ";
		startTime = std::chrono::high_resolution_clock::now();
	};

	void LogStats() {
		if (++FrameCount > RefreshRate)
		{
			// Clear stats for each 100 frame
			FrameCount = 0;
			MaxFPS = -INFINITY;
			MinFPS = INFINITY;
		}
		auto now = std::chrono::high_resolution_clock::now();
		auto FPS = 1.0 / (std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count()) * 1000.0;
		MaxFPS = (FPS > MaxFPS) ? (FPS) : (MaxFPS);
		MinFPS = (MinFPS > FPS) ? (FPS) : (MinFPS);
		nosEngine.WatchLog(Name_FPS.c_str(), std::to_string(FPS).c_str());
		nosEngine.WatchLog(Name_MAX_FPS.c_str(), std::to_string(MaxFPS).c_str());
		nosEngine.WatchLog(Name_MIN_FPS.c_str(), std::to_string(MinFPS).c_str());
		startTime = now;
	}

	float GetMaxFPS() const {
		return MaxFPS;
	}

	float GetMinFPS() const {
		return MinFPS;
	}

private:
	std::string Name_FPS, Name_MAX_FPS, Name_MIN_FPS;
	int FrameCount = 0;
	int RefreshRate = 100;
	std::chrono::steady_clock::time_point startTime;
	float MinFPS = 9999, MaxFPS = -9999, FPS = 0;

};

struct NVVFX_AR_NodeContext : nos::NodeContext {

	NVVFXAppRunner VFX;

	StatsLogger logger;
	nos::fb::UUID NodeID, InputID, OutputID;
	nosResourceShareInfo InputFormatted = {}, InputBuffer = {}, output = {};
	nosResourceShareInfo OutputPre = {};

	std::atomic_bool OutReady = false;
	CUDAVulkanInterop interop;
	int lastWidth, lastHeight;

	NvCVImage nvcv_NV_Input;
	NvCVImage nvcv_NV_Output;
	NvCVImage nvcv_NOS_Input;
	NvCVImage nvcv_NOS_Output;

	std::filesystem::path ModelsPath;
	bool OutputSizeSet = false;

	NVVFX_AR_NodeContext(nos::fb::Node const* node) :NodeContext(node), logger("NVVFX") {
		NodeID = *node->id();
		for (const auto& pin : *node->pins()) {
			if (NSN_In.Compare(pin->name()->c_str()) == 0) {
				InputID = *pin->id();
			}
			if (NSN_Out.Compare(pin->name()->c_str()) == 0) {
				OutputID = *pin->id();
			}
		}
		ModelsPath = NSN_ModelsPath.AsString();
		SetPinOrphanState(InputID, true);
		nosResult res = VFX.CreateArtifactReductionEffect(ModelsPath.string());
		if (res == NOS_RESULT_SUCCESS) {
			SetPinOrphanState(InputID, false);
		}
	}


	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override {
		if (pinName == NSN_In) {
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(value.Data);
			lastWidth = input.Info.Texture.Width;
			lastHeight = input.Info.Texture.Height;

		}
	}

	bool InputSizeChanged() {

	}

	bool RefreshInputFormatted(nosResourceShareInfo* in)
	{
		if (InputFormatted.Memory.Handle == NULL)
			return true;
		bool sizeChanged = (in->Info.Texture.Width != InputFormatted.Info.Texture.Width)
			|| (in->Info.Texture.Height != InputFormatted.Info.Texture.Height);

		
		return sizeChanged;
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* execArgs) {
		nos::NodeExecuteArgs args(execArgs);
		nosResult res = NOS_RESULT_SUCCESS;
		nosResourceShareInfo in = nos::vkss::DeserializeTextureInfo(args[NSN_In].Data->Data);

		PrepareResources(in);

		if (res == NOS_RESULT_SUCCESS) {
			nosCmd cmd = {};
			nosGPUEvent gpuevent = {};
			nosVulkan->Begin("Input DownloadD", &cmd);
			nosVulkan->Copy(cmd, &in, &InputFormatted, 0);
			nosVulkan->End2(cmd, NOS_TRUE, &gpuevent);
			nosVulkan->WaitGpuEvent(&gpuevent, 0);

			res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			res = VFX.RunArtifactReduction(&nvcv_NOS_Input, &nvcv_NOS_Output);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			nosResourceShareInfo out = nos::vkss::DeserializeTextureInfo(args[NSN_Out].Data->Data);
			nosCmd cmd2;
			nosGPUEvent gpuevent2 = {};
			nosVulkan->Begin("NVVFX Upload", &cmd2);
			nosVulkan->Copy(cmd2, &output, &out, 0);
			nosVulkan->End(cmd2, NOS_FALSE);
		}

		return res;
	}

	nosResult PrepareResources(nosResourceShareInfo& in) {
		if (in.Info.Texture.Width == InputFormatted.Info.Texture.Width &&
			in.Info.Texture.Height == InputFormatted.Info.Texture.Height &&
			nvcv_NOS_Output.width == in.Info.Texture.Width) {
			//No change
			return NOS_RESULT_SUCCESS;
		}

		nosResult res = NOS_RESULT_SUCCESS;

		if (InputFormatted.Memory.Handle != NULL) {
			nosVulkan->DestroyResource(&InputFormatted);
		}
		//if (output.Memory.Handle != NULL) {
		//	nosVulkan->DestroyResource(&output);
		//}

		InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		InputFormatted.Info.Texture.Width = in.Info.Texture.Width;
		InputFormatted.Info.Texture.Height = in.Info.Texture.Height;
		InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
		InputFormatted.Info.Texture.Format = NOS_FORMAT_R32G32B32A32_SFLOAT;
		nosVulkan->CreateResource(&InputFormatted);

		res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);

		nosResourceShareInfo dummy = {};
		dummy.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		dummy.Info.Texture.Width = InputFormatted.Info.Texture.Width;
		dummy.Info.Texture.Height = InputFormatted.Info.Texture.Height;
		dummy.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
		dummy.Info.Texture.Format = InputFormatted.Info.Texture.Format;
		dummy.Memory.ExternalMemory.HandleType = NOS_EXTERNAL_MEMORY_HANDLE_TYPE_WIN32;

		auto texFb = nos::vkss::ConvertTextureInfo(dummy);
		texFb.unscaled = true;
		auto texFbBuf = nos::Buffer::From(texFb);
		nosEngine.SetPinValue(OutputID, { .Data = texFbBuf.Data(), .Size = texFbBuf.Size() });

		res = interop.AllocateNVCVImage("Trial", nvcv_NOS_Input.width, nvcv_NOS_Input.height,
			nvcv_NOS_Input.pixelFormat, nvcv_NOS_Input.componentType,
			nvcv_NOS_Input.bufferBytes,
			NVCV_INTERLEAVED, &nvcv_NOS_Output);

		res = interop.NVCVImageToNosTexture(nvcv_NOS_Output, output);

		NvCVImage srcTemp = {}, dstTemp = {};
		res = interop.AllocateNVCVImage("SrcTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_F32, nvcv_NOS_Input.bufferBytes, NVCV_PLANAR, &srcTemp);
		res = interop.AllocateNVCVImage("DstTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_F32,
			nvcv_NOS_Input.bufferBytes, NVCV_PLANAR, &dstTemp);

		res = VFX.InitTransferBuffers(&srcTemp, &dstTemp);

		OutputSizeSet = true;
		return res;
	}

	void SetPinOrphanState(nos::fb::UUID uuid, bool isOrphan) {
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<::flatbuffers::Offset<nos::PartialPinUpdate>> toUpdate;
		toUpdate.push_back(nos::CreatePartialPinUpdateDirect(fbb, &uuid, 0, nos::fb::CreateOrphanStateDirect(fbb, isOrphan)));

		flatbuffers::FlatBufferBuilder fbb2;
		HandleEvent(
			CreateAppEvent(fbb, nos::CreatePartialNodeUpdateDirect(fbb, &NodeID, nos::ClearFlags::NONE, 0, 0, 0, 0, 0, 0, 0, &toUpdate)));
	}
};


void RegisterNVVFX_AR(nosNodeFunctions* outFunctions) {
	outFunctions->ClassName = NSN_NVVFX_AR; outFunctions->OnNodeCreated = [](const nosFbNode* node, void** ctx) { *ctx = new NVVFX_AR_NodeContext(node); }; outFunctions->OnNodeDeleted = [](void* ctx, nosUUID nodeId) { delete static_cast<NVVFX_AR_NodeContext*>(ctx); }; outFunctions->OnNodeUpdated = [](void* ctx, const nosFbNode* updatedNode) { static_cast<NVVFX_AR_NodeContext*>(ctx)->DoOnNodeUpdated(updatedNode); }; 
	outFunctions->OnPinValueChanged = [](void* ctx, nosName pinName, nosUUID pinId, nosBuffer value) { 
		static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPinValueChanged(pinName, pinId, value); 
		}; 
	outFunctions->OnPinConnected = [](void* ctx, nosName pinName, nosUUID connectedPin, nosUUID) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPinConnected(pinName, connectedPin); }; outFunctions->OnPinDisconnected = [](void* ctx, nosName pinName) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPinDisconnected(pinName); }; outFunctions->OnPinShowAsChanged = [](void* ctx, nosName pinName, nosFbShowAs showAs) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPinShowAsChanged(pinName, showAs); }; outFunctions->OnPathCommand = [](void* ctx, const nosPathCommand* command) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPathCommand(command); }; outFunctions->ExecuteNode = [](void* ctx, const nosNodeExecuteArgs* args) { return static_cast<NVVFX_AR_NodeContext*>(ctx)->ExecuteNode(args); }; outFunctions->CopyFrom = [](void* ctx, nosCopyInfo* copyInfo) { return static_cast<NVVFX_AR_NodeContext*>(ctx)->CopyFrom(copyInfo); }; outFunctions->CopyTo = [](void* ctx, nosCopyInfo* copyInfo) { return static_cast<NVVFX_AR_NodeContext*>(ctx)->CopyTo(copyInfo); }; outFunctions->OnMenuRequested = [](void* ctx, const nosContextMenuRequest* request) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnMenuRequested(request); }; outFunctions->OnMenuCommand = [](void* ctx, nosUUID itemID, uint32_t cmd) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnMenuCommand(itemID, cmd); }; outFunctions->OnKeyEvent = [](void* ctx, const nosKeyEvent* keyEvent) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnKeyEvent(keyEvent); }; outFunctions->OnPinDirtied = [](void* ctx, nosUUID pinID, uint64_t frameCount) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPinDirtied(pinID, frameCount); }; outFunctions->OnPathStateChanged = [](void* ctx, nosPathState pathState) { static_cast<NVVFX_AR_NodeContext*>(ctx)->OnPathStateChanged(pathState); }; outFunctions->CanRemoveOrphanPin = [](void* ctx, nosName pinName, nosUUID pinId) { return static_cast<NVVFX_AR_NodeContext*>(ctx)->CanRemoveOrphanPin(pinName, pinId); }; outFunctions->OnOrphanPinRemoved = [](void* ctx, nosName pinName, nosUUID pinId) { return static_cast<NVVFX_AR_NodeContext*>(ctx)->OnOrphanPinRemoved(pinName, pinId); }; outFunctions->CanCreateNode = NVVFX_AR_NodeContext::CanCreateNode; outFunctions->GetFunctions = NVVFX_AR_NodeContext::GetFunctions;;
}