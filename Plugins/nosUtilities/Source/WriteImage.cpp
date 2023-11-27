// Copyright Nodos AS. All Rights Reserved.

#include <Nodos/Helpers.hpp>
#include "Builtins_generated.h"
#include <AppService_generated.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "../Shaders/Linear2SRGB.frag.spv.dat"

#include <mutex>

namespace nos::utilities
{
NOS_REGISTER_NAME(Linear2SRGB_Pass);
NOS_REGISTER_NAME(Linear2SRGB_Shader);
NOS_REGISTER_NAME(Path);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_WriteImage, "nos.utilities.WriteImage")

struct WriteImage : NodeContext {
    std::filesystem::path Path;
    nosResourceShareInfo Input;
    std::atomic_bool WriteRequested = false;
    std::condition_variable CV;
    std::mutex Mutex;
    std::thread Worker;
    std::atomic_bool Write = false;
    std::atomic_bool ShouldStop = false;

    WriteImage(nosFbNode const* node) : NodeContext(node){
        Worker = std::thread([this] {
            while (!ShouldStop) {
                std::unique_lock<std::mutex> lock(Mutex);
                CV.wait(lock, [this] { return Write || ShouldStop; });
                if (ShouldStop)
                    break;
                if (this->Write) {
					this->Write = false;
					this->WriteImageToFile();
				}
            }
        });
        for (auto* pin : *node->pins()) {
            auto* pinData = pin->data();
            nosBuffer value = { .Data = (void*)pinData->data(), .Size = pinData->size() };
            OnPinValueChanged(nosEngine.GetName(pin->name()->c_str()), *pin->id(), &value);
        }
    }

    ~WriteImage() {
		ShouldStop = true;
		CV.notify_all();
		Worker.join();
	}

    void SignalWrite() {
		std::unique_lock<std::mutex> lock(Mutex);
        Write = true;
		CV.notify_all();
	}

    void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer* value) override 
    {
        std::unique_lock<std::mutex> lock(Mutex);
		if (pinName == NSN_In)
			Input = DeserializeTextureInfo(value->Data);
		else if (pinName == NSN_Path)
			Path = std::string((const char*)value->Data, value->Size);
	}

    nosResult BeginCopyTo(nosCopyInfo* copyInfo) override
    {
        copyInfo->ShouldSubmitAndWait = true;
        copyInfo->Stop = true;
        return NOS_RESULT_SUCCESS;
    }

    void EndCopyTo(nosCopyInfo* copyInfo) override
    {
        if (WriteRequested) {
            WriteRequested = false;
            SignalWrite();
        }
    }

    void WriteImageToFile() {
        auto& path = this->Path;
        auto& input = this->Input;
        try {
            if (!std::filesystem::exists(path.parent_path()))
                std::filesystem::create_directories(path.parent_path());
        }
        catch (std::filesystem::filesystem_error& e) {
            nosEngine.LogE("WriteImage - %s: %s", path.string().c_str(), e.what());
            return;
        }
        nosEngine.LogI("WriteImage: Writing frame to file %s", path.string().c_str());

        struct Captures
        {
            nosResourceShareInfo SRGB;
            nosResourceShareInfo Buf = {};
            std::filesystem::path Path;
        } captures = Captures{ .SRGB = input,.Path = path };

        captures.SRGB.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
        captures.SRGB.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
        nosEngine.Create(&captures.SRGB);

        nosCmd cmd;
        nosEngine.Begin(&cmd);
        nosEngine.Copy(cmd, &input, &captures.SRGB, nullptr);
        nosEngine.Download(cmd, &captures.SRGB, &captures.Buf);
        nosEngine.End(cmd);

        if (auto buf2write = nosEngine.Map(&captures.Buf))
            if (!stbi_write_png(captures.Path.string().c_str(), captures.SRGB.Info.Texture.Width, captures.SRGB.Info.Texture.Height, 4, buf2write, captures.SRGB.Info.Texture.Width * 4))
                nosEngine.LogE("WriteImage: Unable to write frame to file", "");
            else
                nosEngine.LogI("WriteImage: Wrote frame to file %s", captures.Path.string().c_str());
        nosEngine.Destroy(&captures.Buf);
        nosEngine.Destroy(&captures.SRGB);
    }

	static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
	{
		*outCount = 1;
		if (!outShaders)
			return NOS_RESULT_SUCCESS;
        outShaders[0] = {.Key = NSN_Linear2SRGB_Shader, .Source = {.SpirvBlob = {(void*)Linear2SRGB_frag_spv, sizeof(Linear2SRGB_frag_spv) }}};
		return NOS_RESULT_SUCCESS;
	}

    static nosResult GetPasses(size_t* count, nosPassInfo* passes)
    {
        *count = 1;
        if (!passes)
            return NOS_RESULT_SUCCESS;

        *passes = nosPassInfo{
            .Key = NSN_Linear2SRGB_Pass,
            .Shader = NSN_Linear2SRGB_Shader,
            .Blend = 0,
            .MultiSample = 1,
        };

        return NOS_RESULT_SUCCESS;
    }

    static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
    {
        *count = 1;
        if (!names || !fns)
            return NOS_RESULT_SUCCESS;

        *names = NOS_NAME_STATIC("WriteImage_Save");
        *fns = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
        {
            auto writeImage = (WriteImage*)ctx;
            auto ids = GetPinIds(nodeArgs);
            writeImage->WriteRequested = true;
            nosEngine.SchedulePin(ids[NSN_In], {0, 1});
            nosEngine.LogI("WriteImage: Write requested");
        };

        return NOS_RESULT_SUCCESS;
    }
};

void RegisterWriteImage(nosNodeFunctions* fn)
{
    NOS_BIND_NODE_CLASS(NSN_Nos_Utilities_WriteImage, WriteImage, fn);
}

} // namespace nos