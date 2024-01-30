#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <nosUtil/Thread.h>
#include <nosVulkanSubsystem/Helpers.hpp>
#include "NVVFXInterop.h"
#include "NVVFXAppRunner.h"
#include "NVVFX_Names.h"
#include "nosCUDASubsystem/nosCUDASubsystem.h"
// nosNodes

struct NVVFX_AIGS_NodeContext : nos::NodeContext {

	NVVFXAppRunner VFX;
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
	nosCUDACallbackContext context = {};

	NVVFX_AIGS_NodeContext(nos::fb::Node const* node) :NodeContext(node) {
		
		nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 1, 0, (void**)&nosVulkan);
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
		nosResult res = VFX.CreateAIGreenScreenEffect(ModelsPath.string());
		if (res == NOS_RESULT_SUCCESS) {
			SetPinOrphanState(InputID, false);
		}
		
		nosCUDAModule normalizeModule = {};
		nosCUDAKernelFunction func = {};
		nosCUDA->LoadKernelModuleFromPTX("D:/CmdCompilation/CommonCUDAKernels.ptx", &normalizeModule);
		nosCUDA->GetModuleKernelFunction("NormalizeKernel", normalizeModule, &func);
		nosCUDAKernelLaunchConfig config = {};
		config.GridDimensions = { 1,1,1 };
		config.BlockDimensions = { 256, 1, 1 };
		config.DynamicMemorySize = 0;
		nosCUDAStream stream = {};
		nosCUDA->CreateStream(&stream);
		int SIZE = 256;
		size_t allocSize = 256 * sizeof(float);
		float MAX = RAND_MAX;
		nosCUDABufferInfo buf = {};
		nosCUDABufferInfo cpuBuf = {};
		nosCUDA->CreateShareableBufferOnCUDA(&buf, allocSize);
		nosCUDA->CreateBuffer(&cpuBuf, allocSize);
		float* dataBuf = reinterpret_cast<float*>(cpuBuf.Address);
		
		for (int i = 0; i < SIZE; i++) {
			dataBuf[i] = std::rand();
		}

		nosCUDA->CopyBuffers(&cpuBuf, &buf);
		float* gpuDataPointer = reinterpret_cast<float*>(buf.Address);
		void* args[] = { &gpuDataPointer, &MAX, &SIZE};
		nosCUDACallbackContext contx = {};
		contx.Data = dataBuf;
		nosCUDA->LaunchModuleKernelFunction(stream, func, config, args, cbFunc, &contx);
		nosCUDA->WaitStream(stream);

		nosCUDABufferInfo resBuf = {};
		nosCUDA->CreateBuffer(&resBuf, allocSize);

		void* devPtr = NULL;


		float* resData = new float[SIZE];

		int a = 5;
	}
	
	static void NOS_CUDA_CALLBACK cbFunc(void* data) {
		nosCUDACallbackContext* ctx = reinterpret_cast<nosCUDACallbackContext*>(data);
		int a = *reinterpret_cast<int*>(ctx->Data);

	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override {
		if (pinName == NSN_In) {
			nosResourceShareInfo input = nos::vkss::DeserializeTextureInfo(value.Data);
			lastWidth = input.Info.Texture.Width;
			lastHeight = input.Info.Texture.Height;
		}
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

		res = PrepareResources(in);

		if (res == NOS_RESULT_SUCCESS) {

			nosCmd cmd = {};
			nosGPUEvent gpuevent = {};
			nosVulkan->Begin("Input Download", &cmd);
			nosVulkan->Copy(cmd, &in, &InputFormatted, 0);
			nosVulkan->End2(cmd, NOS_TRUE, &gpuevent);
			nosVulkan->WaitGpuEvent(&gpuevent, 0);

			res = interop.nosTextureToNVCVImage(InputFormatted, nvcv_NOS_Input);
			if (res != NOS_RESULT_SUCCESS)
				return res;

			res = VFX.RunAIGreenScreenEffect(&nvcv_NOS_Input, &nvcv_NOS_Output);
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

		InputFormatted.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		InputFormatted.Info.Texture.Width = in.Info.Texture.Width;
		InputFormatted.Info.Texture.Height = in.Info.Texture.Height;
		InputFormatted.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_DST | NOS_IMAGE_USAGE_TRANSFER_SRC);
		InputFormatted.Info.Texture.Format = NOS_FORMAT_B8G8R8A8_UNORM;
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
		res = interop.AllocateNVCVImage("SrcTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_BGR, NVCV_U8, nvcv_NOS_Input.bufferBytes, NVCV_CHUNKY, &srcTemp);
		res = interop.AllocateNVCVImage("DstTemp", nvcv_NOS_Input.width, nvcv_NOS_Input.height, NVCV_A, NVCV_U8,
			nvcv_NOS_Input.bufferBytes, NVCV_CHUNKY, &dstTemp);

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


void RegisterNVVFX_AIGreenScreen(nosNodeFunctions* outFunctions) {
	NOS_BIND_NODE_CLASS(NSN_NVVFX_AIGreenScreen, NVVFX_AIGS_NodeContext, outFunctions);
}