// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>
#include "Builtins_generated.h"
#include <AppService_generated.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "../Shaders/Linear2SRGB.frag.spv.dat"

#include <mutex>

namespace mz::utilities
{
MZ_REGISTER_NAME(Linear2SRGB_Pass);
MZ_REGISTER_NAME(Linear2SRGB_Shader);
MZ_REGISTER_NAME(Path);
MZ_REGISTER_NAME(In);
MZ_REGISTER_NAME_SPACED(Mz_Utilities_WriteImage, "mz.utilities.WriteImage")

struct WriteImage {
    std::filesystem::path Path;
    mzResourceShareInfo Input;
    std::atomic_bool WriteRequested = false;
    std::condition_variable CV;
    std::mutex Mutex;
    std::thread Worker;
    std::atomic_bool Write = false;
    std::atomic_bool ShouldStop = false;

    WriteImage() {
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

    void OnPinValueChanged(mzName pinName, mzBuffer* value) {
        std::unique_lock<std::mutex> lock(Mutex);
		if (pinName == MZN_In)
			Input = DeserializeTextureInfo(value->Data);
		else if (pinName == MZN_Path)
			Path = std::string((const char*)value->Data, value->Size);
	}

    void WriteImageToFile() {
        auto& path = this->Path;
        auto& input = this->Input;
        try {
            if (!std::filesystem::exists(path.parent_path()))
                std::filesystem::create_directories(path.parent_path());
        }
        catch (std::filesystem::filesystem_error& e) {
            mzEngine.LogE("WriteImage - %s: %s", path.string().c_str(), e.what());
            return;
        }
        mzEngine.LogI("WriteImage: Writing frame to file %s", path.string().c_str());

        struct Captures
        {
            mzResourceShareInfo SRGB;
            mzResourceShareInfo Buf = {};
            std::filesystem::path Path;
        } captures = Captures{ .SRGB = input,.Path = path };

        captures.SRGB.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
        captures.SRGB.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
        mzEngine.Create(&captures.SRGB);

        mzCmd cmd;
        mzEngine.Begin(&cmd);
        mzEngine.Blit(cmd, &input, &captures.SRGB);
        mzEngine.Download(cmd, &captures.SRGB, &captures.Buf);
        mzEngine.End(cmd);

        if (auto buf2write = mzEngine.Map(&captures.Buf))
            if (!stbi_write_png(captures.Path.string().c_str(), captures.SRGB.Info.Texture.Width, captures.SRGB.Info.Texture.Height, 4, buf2write, captures.SRGB.Info.Texture.Width * 4))
                mzEngine.LogE("WriteImage: Unable to write frame to file", "");
            else
                mzEngine.LogI("WriteImage: Wrote frame to file %s", captures.Path.string().c_str());
        mzEngine.Destroy(&captures.Buf);
        mzEngine.Destroy(&captures.SRGB);
    }
};

mzResult GetShaders(size_t* count, mzName* names, mzBuffer* spirv)
{
    *count = 1;
    if (!names || !spirv)
        return MZ_RESULT_SUCCESS;

    *names = MZN_Linear2SRGB_Shader;
    spirv->Data = (void*)Linear2SRGB_frag_spv;
    spirv->Size = sizeof(Linear2SRGB_frag_spv);
    return MZ_RESULT_SUCCESS;
}

static mzResult GetPasses(size_t* count, mzPassInfo* passes)
{
    *count = 1;
    if (!passes)
        return MZ_RESULT_SUCCESS;

    *passes = mzPassInfo{
        .Key = MZN_Linear2SRGB_Pass,
        .Shader = MZN_Linear2SRGB_Shader,
        .Blend = 0,
        .MultiSample = 1,
    };

    return MZ_RESULT_SUCCESS;
}

static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns)
{
    *count = 1;
    if(!names || !fns)
        return MZ_RESULT_SUCCESS;

    *names = MZ_NAME_STATIC("WriteImage_Save");
    *fns = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs)
    {
        auto writeImage = (WriteImage*)ctx;
        auto ids = GetPinIds(nodeArgs);
        writeImage->WriteRequested = true;
        mzEngine.SchedulePin(ids[MZN_In]);
        mzEngine.LogI("WriteImage: Write requested");
    };
    
    return MZ_RESULT_SUCCESS;
}

void RegisterWriteImage(mzNodeFunctions* fn)
{
	fn->TypeName = MZN_Mz_Utilities_WriteImage;
    fn->GetFunctions = GetFunctions;
    fn->GetShaders = GetShaders;
    fn->GetPasses = GetPasses;
    fn->OnPinValueChanged = [](void* ctx, mzName pinName, mzBuffer* value) {
		auto writeImage = (WriteImage*)ctx;
        writeImage->OnPinValueChanged(pinName, value);
	};
    fn->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr)
	{
        auto* writeImage = new WriteImage();
	    *outCtxPtr = writeImage;
        for (auto* pin : *node->pins()) {
            auto* pinData = pin->data();
            mzBuffer value = {.Data = (void*)pinData->data(), .Size = pinData->size() };
            writeImage->OnPinValueChanged(mzEngine.GetName(pin->name()->c_str()), &value);
        }
    };
    fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) {
        delete (WriteImage*)ctx;
	};
    fn->BeginCopyTo = [](void* ctx, mzCopyInfo* copyInfo) {
        copyInfo->ShouldSubmitAndWait = true;
        copyInfo->Stop = true;
        return MZ_RESULT_SUCCESS;
	};
    fn->EndCopyTo = [](void* ctx, mzCopyInfo* copyInfo) {
        auto* writeImage = (WriteImage*)ctx;
        if (writeImage->WriteRequested) {
			writeImage->WriteRequested = false;
			writeImage->SignalWrite();
		}
    };
}

} // namespace mz