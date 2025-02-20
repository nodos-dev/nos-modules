// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include "Builtins_generated.h"
#include <AppService_generated.h>
#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <nosVulkanSubsystem/Helpers.hpp>

#include <mutex>

#include "Names.h"

namespace nos::utilities
{
NOS_REGISTER_NAME(IncludeAlpha);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_WriteImage, "nos.utilities.WriteImage")

struct WriteImage : NodeContext {
    std::filesystem::path Path;
    std::optional<vkss::Resource> TempSrgbCopy;
    bool IncludeAlpha = false;
    nosGPUEvent Event = 0;
    std::atomic_bool WriteRequested = false;
    std::condition_variable CV;
    std::mutex Mutex;
    std::thread Worker;
    std::atomic_bool Write = false;
    std::atomic_bool ShouldStop = false;

    WriteImage(nosFbNodePtr node) : NodeContext(node){
        Worker = std::thread([this] {
            while (!ShouldStop) {
                std::unique_lock<std::mutex> lock(Mutex);
                CV.wait(lock, [this] { return Write || ShouldStop; });
                if (ShouldStop)
                    break;
                if(Event)
                    nosVulkan->WaitGpuEvent(&Event, UINT64_MAX);
                if (this->Write) {
					this->Write = false;
					this->WriteImageToFile();
				}
            }
        });
        for (auto* pin : *node->pins()) {
            auto* pinData = pin->data();
            nosBuffer value = { .Data = (void*)pinData->data(), .Size = pinData->size() };
            OnPinValueChanged(nos::Name(pin->name()->c_str()), *pin->id(), value);
        }
    }

    ~WriteImage() {
		ShouldStop = true;
		CV.notify_all();
		Worker.join();
	}

    void SendWriteRequest(nosFunctionExecuteParams* params)
    {
        nosEngine.LogI("WriteImage: Write requested");

        nos::NodeExecuteParams execParams(params->FunctionNodeExecuteParams);
        
        std::unique_lock<std::mutex> lock(Mutex);
        Path = std::utf8_to_wstring(std::string((const char*)execParams[NSN_Path].Data->Data, execParams[NSN_Path].Data->Size));
        IncludeAlpha = *(bool*)execParams[NSN_IncludeAlpha].Data->Data;
        assert(Event == 0);
        nosCmd cmd = vkss::BeginCmd(NOS_NAME("Write Image Copy To"), NodeId);
        auto inputTex = vkss::DeserializeTextureInfo(execParams[NSN_In].Data->Data);
        TempSrgbCopy = {};

        nosResourceShareInfo tempSrgbInfo = inputTex;
		tempSrgbInfo.Info.Texture.Format = NOS_FORMAT_R8G8B8A8_SRGB;
		tempSrgbInfo.Info.Texture.Usage = nosImageUsage(NOS_IMAGE_USAGE_TRANSFER_SRC | NOS_IMAGE_USAGE_TRANSFER_DST);
		TempSrgbCopy = vkss::Resource::Create(tempSrgbInfo, "TempSrgbCopy");
        
        nosVulkan->Copy(cmd, &inputTex, &*TempSrgbCopy, 0);
        nosCmdEndParams endParams{ .ForceSubmit = true, .OutGPUEventHandle = &Event };
        nosVulkan->End(cmd, &endParams);
        Write = true;
        CV.notify_all();
    }

    void WriteImageToFile() {
        std::string UTF8Path = std::wstring_to_utf8(this->Path);
        try {
            if (!std::filesystem::exists(this->Path.parent_path()))
                std::filesystem::create_directories(this->Path.parent_path());
        }
        catch (std::filesystem::filesystem_error& e) {
            nosEngine.LogE("WriteImage - %s: %s", UTF8Path.c_str(), e.what());
            return;
        }
        nosEngine.LogI("WriteImage: Writing frame to file %s", UTF8Path.c_str());

		auto optBuf = vkss::Resource::DownloadTextureSync(*TempSrgbCopy, "TempSrgbCopy Downloaded");
        if (!optBuf) {
            nosEngine.LogE("WriteImage: Unable to download frame");
            return;
        }
        auto buf = std::move(*optBuf);

        if (auto buf2write = nosVulkan->Map(&buf))
        {
            if(!IncludeAlpha)
                for (size_t i = 3; i < buf.Info.Buffer.Size; i += 4)
                    buf2write[i] = 0xff;

            if (!stbi_write_png(UTF8Path.c_str(), TempSrgbCopy->Info.Texture.Width, TempSrgbCopy->Info.Texture.Height, 4, buf2write, TempSrgbCopy->Info.Texture.Width * 4))
                nosEngine.LogE("WriteImage: Unable to write frame to file", "");
            else
                nosEngine.LogI("WriteImage: Wrote frame to file %s", UTF8Path.c_str());
        }
        TempSrgbCopy = {};
    }

    static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
    {
        *count = 1;
        if (!names || !fns)
            return NOS_RESULT_SUCCESS;

        *names = NOS_NAME_STATIC("WriteImage_Save");
        *fns = [](void* ctx, nosFunctionExecuteParams* params)
        {
            auto writeImage = (WriteImage*)ctx;
            writeImage->SendWriteRequest(params);
            return NOS_RESULT_SUCCESS;
        };

        return NOS_RESULT_SUCCESS;
    }
};

nosResult RegisterWriteImage(nosNodeFunctions* fn)
{
    NOS_BIND_NODE_CLASS(NSN_Nos_Utilities_WriteImage, WriteImage, fn);
    return NOS_RESULT_SUCCESS;
}

} // namespace nos