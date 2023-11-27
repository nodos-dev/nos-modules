#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/Helpers.hpp>
#include <AppService_generated.h>
#include <Windows.h>
#include <nosUtil/Thread.h>
// nosNodes
#
#include "../Shaders/SRGB2Linear.frag.spv.dat"
#include <stb_image.h>
#include <stb_image_write.h>

NOS_INIT();
NOS_REGISTER_NAME(In)
NOS_REGISTER_NAME(DLLPath)
NOS_REGISTER_NAME(ParamPath)
NOS_REGISTER_NAME(BinPath)
NOS_REGISTER_NAME(Out)
NOS_REGISTER_NAME(SRGB2Linear_Pass);
NOS_REGISTER_NAME(SRGB2Linear_Shader);
NOS_REGISTER_NAME(CNN);


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
	nosResourceShareInfo Input;
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
		Input.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
		Input.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
	}

	nosResult LoadFromPath(const char* modelPath, const char* paramPath, const char* binPath) {
		FreeLibrary(cnnDLL);
		cnnDLL = LoadLibraryExA(modelPath, nullptr, 0);
		std::cout << "Trying to load library from path: " << modelPath << std::endl;
		if (cnnDLL == 0) {
			std::cout << "Library load failed" << std::endl;
			return NOS_RESULT_FAILED;
		}
		detectFunction = reinterpret_cast<detectFuncType>(GetProcAddress(cnnDLL, "ImageDetection"));
		initFunction = reinterpret_cast<InitFuncType>(GetProcAddress(cnnDLL, "InitNet"));
		initFunction(paramPath, binPath);
		if (initFunction && detectFunction)
			nosEngine.LogI("Model succesfully loaded");
		else
			nosEngine.LogE("Incorrect model parameters!");
		return NOS_RESULT_SUCCESS;
	}

	~SimpleDLLLoader() {
		FreeLibrary(cnnDLL);
	}
};
SimpleDLLLoader dllLoader;


struct CNNThread : public nos::Thread{
	uint8_t* data = nullptr;
	size_t size = 0;
	nosResourceShareInfo Buf = {};
	nosResourceShareInfo Out = {};
	std::atomic<bool> isDone = true;
	std::atomic<bool> isNewFrameReady = false;
	std::atomic<bool> isNOSHungry = true;
	nosResourceShareInfo InputRGBA8 = {};

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
			nosCmd cmd;
			nosEngine.Begin(&cmd);
			nosEngine.Copy(cmd, &InputRGBA8, &Buf, 0);
			nosEngine.End(cmd);
			
			auto t1 = std::chrono::high_resolution_clock::now();

			nosEngine.WatchLog("CNN Download Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			auto buf2write = nosEngine.Map(&Buf);
			t0 = std::chrono::high_resolution_clock::now();
			memcpy(data, buf2write, size);
			t1 = std::chrono::high_resolution_clock::now();
			nosEngine.WatchLog("CNN Memcpy Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()).c_str());

			t1 = std::chrono::high_resolution_clock::now();
			if (dllLoader.detectFunction != nullptr && data != nullptr)
			{
				dllLoader.detectFunction(data, InputRGBA8.Info.Texture.Width, InputRGBA8.Info.Texture.Height);
			}

			auto t2 = std::chrono::high_resolution_clock::now();
			nosEngine.WatchLog("CNN Detection Time (us)", std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()).c_str());

			isDone = true;

		}
	}
};

struct CNNNodeContext :  nos::NodeContext {
	CNNThread Worker;
	nosResourceShareInfo TempOutput = {};
	//On Node Created
	CNNNodeContext(nos::fb::Node const* node):NodeContext(node) {
		
		TempOutput.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		TempOutput.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
		TempOutput.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);

		Worker.InputRGBA8.Info.Texture.Width = 1920;
		Worker.InputRGBA8.Info.Texture.Height = 1080;

		Worker.InputRGBA8.Info.Type = NOS_RESOURCE_TYPE_TEXTURE;
		Worker.InputRGBA8.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
		Worker.InputRGBA8.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
		
		nosEngine.Create(&Worker.InputRGBA8);

		Worker.Buf.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		Worker.size = Worker.InputRGBA8.Info.Texture.Width * Worker.InputRGBA8.Info.Texture.Height * 4 * sizeof(uint8_t);
		Worker.Buf.Info.Buffer.Size = Worker.size;
		Worker.Buf.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_TRANSFER_SRC | NOS_BUFFER_USAGE_TRANSFER_DST);
		Worker.data = new uint8_t[Worker.size];
		nosEngine.Create(&Worker.Buf);

		Worker.Start();
	}

	~CNNNodeContext() override {
		Worker.Stop();
		nosEngine.Destroy(&Worker.InputRGBA8);
		nosEngine.Destroy(&Worker.Buf);
		delete[] Worker.data;
	}

	void  OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) override {
		if (dllLoader.cnnDLL == 0) {
			nosEngine.LogE("Missing CNN dll, no execution will be performed.");
			return;
		}
		if (pinName == NSN_In) {

			dllLoader.Input = nos::DeserializeTextureInfo(value->Data);
			nosEngine.LogI("Input value on CNN changed!");
		}
	}

	virtual nosResult ExecuteNode(const nosNodeExecuteArgs* args) override { 
		if (dllLoader.cnnDLL == 0) {
			nosEngine.LogE("Missing CNN dll, no execution will be performed.");
			return NOS_RESULT_FAILED;
		}
		auto values = nos::GetPinValues(args);


		if (Worker.isDone) {
			Worker.isNewFrameReady = true;
			nosResourceShareInfo out = nos::DeserializeTextureInfo(values[NSN_Out]);;
			nosResourceShareInfo tmp = out;
			nosEngine.ImageLoad(Worker.data,
				nosVec2u(out.Info.Texture.Width, out.Info.Texture.Height),
				NOS_FORMAT_R8G8B8A8_SRGB, &tmp);
			{
				nosCmd cmd;
				nosEngine.Begin(&cmd);
				nosEngine.Copy(cmd, &tmp, &out, 0);
				nosEngine.End(cmd);
				nosEngine.Destroy(&tmp);
			}
			//return NOS_RESULT_SUCCESS;
		}

		//return NOS_RESULT_FAILED;
		return NOS_RESULT_SUCCESS;
	}

	nosResult BeginCopyTo(nosCopyInfo* cpy) override {
		cpy->ShouldCopyTexture = true;
		cpy->CopyTextureFrom = dllLoader.Input;
		cpy->CopyTextureTo = Worker.InputRGBA8;
		cpy->ShouldSubmitAndWait = true;
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns) {
		*count = 1;
		if (!names || !fns)
			return NOS_RESULT_SUCCESS;

		*names = NOS_NAME_STATIC("CNNPlugin_LoadModel");
		*fns = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs) {
			auto values = nos::GetPinValues(nodeArgs);
			std::filesystem::path dllPath = GetPinValue<const char>(values, NSN_DLLPath);
			std::filesystem::path paramPath = GetPinValue<const char>(values, NSN_ParamPath);
			std::filesystem::path binPath = GetPinValue<const char>(values, NSN_BinPath);
			if (!std::filesystem::exists(dllPath) || !std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath)) {
				nosEngine.LogE("CNN Plugin cannot load model");
				return;
			}
			dllLoader.LoadFromPath(dllPath.string().c_str(), paramPath.string().c_str(), binPath.string().c_str());
			};

		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders) {
		*outCount = 1;
		if (!outShaders)
			return NOS_RESULT_SUCCESS;

		outShaders[0] = { .Key = NSN_SRGB2Linear_Shader, .Source = {.SpirvBlob = {(void*)SRGB2Linear_frag_spv, sizeof(SRGB2Linear_frag_spv)}} };
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetPasses(size_t* count, nosPassInfo* passes) {
		*count = 1;
		if (!passes)
			return NOS_RESULT_SUCCESS;

		*passes = nosPassInfo{
			.Key = NSN_SRGB2Linear_Pass,
			.Shader = NSN_SRGB2Linear_Shader,
			.Blend = 0,
			.MultiSample = 1,
		};

		return NOS_RESULT_SUCCESS;
	}
};

extern "C"
{

	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions) {
		*outCount = (size_t)(1);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		NOS_BIND_NODE_CLASS(NSN_CNN, CNNNodeContext, outFunctions[0]);


		return NOS_RESULT_SUCCESS;
	}
}

