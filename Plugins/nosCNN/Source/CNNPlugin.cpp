#include <MediaZ/PluginAPI.h>
#include <Builtins_generated.h>
#include <MediaZ/Helpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <mzUtil/Thread.h>
// mzNodes
#
#include "../Shaders/SRGB2Linear.frag.spv.dat"
#include <stb_image.h>
#include <stb_image_write.h>

MZ_INIT();
MZ_REGISTER_NAME(In)
MZ_REGISTER_NAME(DLLPath)
MZ_REGISTER_NAME(ParamPath)
MZ_REGISTER_NAME(BinPath)
MZ_REGISTER_NAME(Out)
MZ_REGISTER_NAME(SRGB2Linear_Pass);
MZ_REGISTER_NAME(SRGB2Linear_Shader);
MZ_REGISTER_NAME(CNN);


//TODO: Do not use globals
//struct CNNObject
//{
//	std::string Name;
//	float Prob;
//	float XCoord, YCoord, Width, Height;
//};

using detectFuncType = void* (__cdecl*)(unsigned char*, int width, int height);
using InitFuncType = void* (__cdecl*)(const char*, const char*);



struct SimpleDLLLoader {

public:
	std::shared_ptr<unsigned char> data;
	HMODULE cnnDLL = 0;
	detectFuncType detectFunction = 0;
	InitFuncType initFunction = 0;
	mzResourceShareInfo Input;
	SimpleDLLLoader() {
		cnnDLL = LoadLibraryExW
		(L"C:\\ncnn\\protobuf_build\\examples\\Debug\\nanodet.dll", nullptr, 0);
		if (cnnDLL != 0) {
			detectFunction = reinterpret_cast<detectFuncType>(GetProcAddress(cnnDLL, "ImageDetection"));
			if (detectFunction != 0)
				std::cout << "Detect function found" << std::endl;
			else
				std::cout << "Detect function NOT found!" << std::endl;

			initFunction = reinterpret_cast<InitFuncType>(GetProcAddress(cnnDLL, "InitNet"));

			if (initFunction != 0) 
				std::cout << "Init function found" << std::endl;
			else
				std::cout << "Init function NOT found!" << std::endl;

		}
		Input.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		Input.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
	}

	mzResult LoadFromPath(const char* modelPath, const char* paramPath, const char* binPath) {
		FreeLibrary(cnnDLL);
		cnnDLL = LoadLibraryExA(modelPath, nullptr, 0);
		std::cout << "Trying to load library from path: " << modelPath << std::endl;
		if (cnnDLL == 0) {
			std::cout << "Library load failed" << std::endl;
			return MZ_RESULT_FAILED;
		}
		detectFunction = reinterpret_cast<detectFuncType>(GetProcAddress(cnnDLL, "ImageDetection"));
		initFunction = reinterpret_cast<InitFuncType>(GetProcAddress(cnnDLL, "InitNet"));
		initFunction(paramPath, binPath);
		if (initFunction && detectFunction)
			mzEngine.LogI("Model succesfully loaded");
		else
			mzEngine.LogE("Incorrect model parameters!");
		return MZ_RESULT_SUCCESS;
	}

	~SimpleDLLLoader() {
		FreeLibrary(cnnDLL);
	}
};
SimpleDLLLoader dllLoader;


struct CNNThread : public mz::Thread{
	uint8_t* data = nullptr;
	size_t size = 0;
	mzResourceShareInfo Buf = {};
	mzResourceShareInfo Out = {};
	std::atomic<bool> isDone = true;
	std::atomic<bool> isNewFrameReady = false;
	std::atomic<bool> isMZHungry = true;
	mzResourceShareInfo InputRGBA8 = {};

	// Inherited via Thread
	void Run() override
	{

		while (!ShouldStop)
		{
			if (!isNewFrameReady) {
				continue;
			}
			isNewFrameReady = false;
			isDone = false;

			auto t0 = std::chrono::high_resolution_clock::now();
			mzCmd cmd;
			mzEngine.Begin(&cmd);
			mzEngine.Copy(cmd, &InputRGBA8, &Buf, 0);
			mzEngine.End(cmd);
			
			auto t1 = std::chrono::high_resolution_clock::now();

			mzEngine.WatchLog("CNN Download Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			auto buf2write = mzEngine.Map(&Buf);
			t0 = std::chrono::high_resolution_clock::now();
			memcpy(data, buf2write, size);
			t1 = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("CNN Memcpy Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			t1 = std::chrono::high_resolution_clock::now();
			if (dllLoader.detectFunction != nullptr && data != nullptr)
			{
				dllLoader.detectFunction(data, InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height);
			}

			auto t2 = std::chrono::high_resolution_clock::now();
			mzEngine.WatchLog("CNN Detection Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()).c_str());

			isDone = true;

		}
	}
};

struct CNNNodeContext :  mz::NodeContext {
	CNNThread Worker;
	mzResourceShareInfo TempOutput = {};
	//On Node Created
	CNNNodeContext(mz::fb::Node const* node):NodeContext(node) {
		
		TempOutput.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		TempOutput.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		TempOutput.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);

		Worker.InputRGBA8.Info.Texture.Width = 1920;
		Worker.InputRGBA8.Info.Texture.Height = 1080;

		Worker.InputRGBA8.Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
		Worker.InputRGBA8.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
		Worker.InputRGBA8.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
		
		mzEngine.Create(&Worker.InputRGBA8);

		Worker.Buf.Info.Type = MZ_RESOURCE_TYPE_BUFFER;
		Worker.size = Worker.InputRGBA8.Info.Texture.Width * Worker.InputRGBA8.Info.Texture.Height * 4 * sizeof(uint8_t);
		Worker.Buf.Info.Buffer.Size = Worker.size;
		Worker.Buf.Info.Buffer.Usage = mzBufferUsage(MZ_BUFFER_USAGE_TRANSFER_SRC | MZ_BUFFER_USAGE_TRANSFER_DST);
		Worker.data = new uint8_t[Worker.size];
		mzEngine.Create(&Worker.Buf);

		Worker.Start();
	}

	~CNNNodeContext() override {
		Worker.Stop();
		mzEngine.Destroy(&Worker.InputRGBA8);
		mzEngine.Destroy(&Worker.Buf);
		delete[] Worker.data;
	}

	void  OnPinValueChanged(mz::Name pinName, mzUUID pinId, mzBuffer* value) override {
		if (dllLoader.cnnDLL == 0) {
			mzEngine.LogE("Missing CNN dll, no execution will be performed.");
			return;
		}
		if (pinName == MZN_In) {

			dllLoader.Input = mz::DeserializeTextureInfo(value->Data);
			mzEngine.LogI("Input value on CNN changed!");
		}
	}

	virtual mzResult ExecuteNode(const mzNodeExecuteArgs* args) override { 
		if (dllLoader.cnnDLL == 0) {
			mzEngine.LogE("Missing CNN dll, no execution will be performed.");
			return MZ_RESULT_FAILED;
		}
		auto values = mz::GetPinValues(args);


		if (Worker.isDone) {
			Worker.isNewFrameReady = true;
			mzResourceShareInfo out = mz::DeserializeTextureInfo(values[MZN_Out]);;
			mzResourceShareInfo tmp = out;
			mzEngine.ImageLoad(Worker.data,
				mzVec2u(out.Info.Texture.Width, out.Info.Texture.Height),
				MZ_FORMAT_R8G8B8A8_SRGB, &tmp);
			{
				mzCmd cmd;
				mzEngine.Begin(&cmd);
				mzEngine.Copy(cmd, &tmp, &out, 0);
				mzEngine.End(cmd);
				mzEngine.Destroy(&tmp);
			}
			//return MZ_RESULT_SUCCESS;
		}

		//return MZ_RESULT_FAILED;
		return MZ_RESULT_SUCCESS;
	}

	mzResult BeginCopyTo(mzCopyInfo* cpy) override {
		cpy->ShouldCopyTexture = true;
		cpy->CopyTextureFrom = dllLoader.Input;
		cpy->CopyTextureTo = Worker.InputRGBA8;
		cpy->ShouldSubmitAndWait = true;
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns) {
		*count = 1;
		if (!names || !fns)
			return MZ_RESULT_SUCCESS;

		*names = MZ_NAME_STATIC("CNNPlugin_LoadModel");
		*fns = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs) {
			auto values = mz::GetPinValues(nodeArgs);
			std::filesystem::path dllPath = GetPinValue<const char>(values, MZN_DLLPath);
			std::filesystem::path paramPath = GetPinValue<const char>(values, MZN_ParamPath);
			std::filesystem::path binPath = GetPinValue<const char>(values, MZN_BinPath);
			if (!std::filesystem::exists(dllPath) || !std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath)) {
				mzEngine.LogE("CNN Plugin cannot load model");
				return;
			}
			dllLoader.LoadFromPath(dllPath.string().c_str(), paramPath.string().c_str(), binPath.string().c_str());
			};

		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetShaders(size_t* outCount, mzShaderInfo* outShaders) {
		*outCount = 1;
		if (!outShaders)
			return MZ_RESULT_SUCCESS;

		outShaders[0] = { .Key = MZN_SRGB2Linear_Shader, .Source = {.SpirvBlob = {(void*)SRGB2Linear_frag_spv, sizeof(SRGB2Linear_frag_spv)}} };
		return MZ_RESULT_SUCCESS;
	}

	static mzResult GetPasses(size_t* count, mzPassInfo* passes) {
		*count = 1;
		if (!passes)
			return MZ_RESULT_SUCCESS;

		*passes = mzPassInfo{
			.Key = MZN_SRGB2Linear_Pass,
			.Shader = MZN_SRGB2Linear_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};

		return MZ_RESULT_SUCCESS;
	}
};

extern "C"
{

	MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outCount, mzNodeFunctions** outFunctions) {
		*outCount = (size_t)(1);
		if (!outFunctions)
			return MZ_RESULT_SUCCESS;

		MZ_BIND_NODE_CLASS(MZN_CNN, CNNNodeContext, outFunctions[0]);


		return MZ_RESULT_SUCCESS;
	}
}

