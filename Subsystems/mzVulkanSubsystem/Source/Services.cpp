#include "Services.h"

// Subsystem
#include "RenderThread.h"
#include "Resources/ResourceManager.h"
#include "Globals.h"

// Vulkan base
#include <mzVulkan/Device.h>
#include <mzVulkan/Command.h>
#include <mzVulkan/Buffer.h>
#include <mzVulkan/Image.h>
#include <mzVulkan/QueryPool.h>

// std
#include <future>
#include <chrono>

extern mzEngineServices mzEngine;

namespace mz::vkss
{
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::string msgType;

	if (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT & messageType)
		msgType += "[General]";
	if (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT & messageType)
		msgType += "[Validation]";
	if (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT & messageType)
		msgType += "[Performance]";

	if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT & messageSeverity)
		mzEngine.LogDD(pCallbackData->pMessage, "%s%s", msgType.c_str(), pCallbackData->pMessageIdName);

	if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT & messageSeverity)
		mzEngine.LogDI(pCallbackData->pMessage, "%s%s", msgType.c_str(), pCallbackData->pMessageIdName);

	if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT & messageSeverity)
		mzEngine.LogDW(pCallbackData->pMessage, "%s%s", msgType.c_str(), pCallbackData->pMessageIdName);

	if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT & messageSeverity)
		mzEngine.LogDE(pCallbackData->pMessage, "%s%s", msgType.c_str(), pCallbackData->pMessageIdName);

	return VK_FALSE;
}

mz::vk::Device* GVkDevice;
vk::rc<mz::vk::Context> GVkCtx;

mzResult Initialize()
{
	GVkCtx = vk::Context::New(DebugCallback);
	if (GVkCtx->Devices.empty())
	{
        GVkCtx = nullptr;
        return MZ_RESULT_FAILED;
	}
	GVkDevice = GVkCtx->Devices[0].get();
	GResources = std::make_unique<ResourceManager>();
	RenderThread::Create();
	Use<RenderThread>()->Start();

    return MZ_RESULT_SUCCESS;
}

mzResult Deinitialize()
{
	Use<RenderThread>()->Stop();
	RenderThread::Destroy();
	GVkDevice = nullptr;
	GVkCtx = nullptr;
	GResources.reset();
    return MZ_RESULT_SUCCESS;
}

static vk::rc<vk::CommandBuffer> CommandStart(mzCmd cmd)
{
    if(!cmd)
    {
        if (mzEngine.IsInSchedulerThread())
            assert(false); // no one in scheduler thread can have the holy cmd
        return GVkDevice->GetPool()->BeginCmd();
    }
    return ((vk::CommandBuffer*)cmd)->shared_from_this();
}

static void CommandFinish(mzCmd inCmd, vk::rc<vk::CommandBuffer> cmd)
{
    if(!inCmd)
    {
        if(mzEngine.IsInSchedulerThread())
            return;
        cmd->Submit();
        cmd->Wait();
    }
}

bool IsMemoryValid(const mzMemoryInfo* mem)
{
	// handle is commented out because the texture created from external sources (i.e. Unreal Engine) comes without a
	// handle handle is created while creating the texture's image only on the first time
	return /*tex->handle &&*/ mem->PID && mem->Memory;
}

static vk::rc<vk::Buffer> ImportBufferDef(vk::Device* vk, const mzResourceShareInfo* buf)
{
	if (!(buf->Memory.PID && buf->Memory.Handle))
	{
		mzEngine.LogE("Trying to import a buffer with invalid parameters. Possibly corrupted flatbuffer");
		return 0;
	}

	if (buf->Memory.PID == mz::vk::PlatformGetCurrentProcessId())
	{
		return ((mz::vk::Buffer*)buf->Memory.Handle)->shared_from_this();
	}

	assert(false && "Unimplemented");
	return 0;
}

static vk::rc<vk::Image> ImportImageDef(vk::Device* vk, const mzResourceShareInfo* tex)
{
	if (!IsMemoryValid(&tex->Memory))
	{
		mzEngine.LogE("Trying to import a buffer with invalid parameters. Possibly corrupted flatbuffer");
		return 0;
	}

	if (tex->Memory.PID == mz::vk::PlatformGetCurrentProcessId() || tex->Memory.Handle)
	{
		if (mzEngine.IsInSchedulerThread())
		{
			// TODO: find a way to do this
			if (!GResources->Exists((vk::Image*)tex->Memory.Handle))
			{
				mzEngine.LogE("Trying to access a deleted texture resource");
				return 0;
			}
		}
		return ((mz::vk::Image*)tex->Memory.Handle)->shared_from_this();
	}

	mz::vk::MemoryExportInfo Ext = {
		.PID = tex->Memory.PID,
		.Memory = (HANDLE)tex->Memory.Memory,
		.Type = (VkExternalMemoryHandleTypeFlags)tex->Memory.Type,
		.Offset = tex->Memory.Offset,
	};

	mz::vk::ImageCreateInfo createInfo = {
		.Extent = {tex->Info.Texture.Width, tex->Info.Texture.Height},
		.Format = (VkFormat)tex->Info.Texture.Format,
		.Usage = (VkImageUsageFlags)tex->Info.Texture.Usage,
		.Type = (VkExternalMemoryHandleTypeFlags)tex->Memory.Type,
		.Imported = &Ext,
	};

	VkResult re;
	auto img = mz::vk::Image::New(vk, createInfo, &re, (HANDLE)tex->Info.Texture.Semaphore);
	if (MZ_VULKAN_FAILED(re))
	{
		mzEngine.LogE("Failed to import image: %s", +::mz::vk::vk_result_string(re));
		return 0;
	}
	return img;
}

mzResult MZAPI_CALL Begin(mzCmd* outCmd)
{ 
    if (mzEngine.IsInSchedulerThread())
    {
        *outCmd = 0;
        return MZ_RESULT_SUCCESS;
    }
    *(vk::CommandBuffer**)outCmd = CommandStart(0).get();
    return MZ_RESULT_SUCCESS; 
}

mzResult MZAPI_CALL End(mzCmd cmd) 
{ 
    if (mzEngine.IsInSchedulerThread())
    {
        return MZ_RESULT_SUCCESS;
    }
    CommandFinish(0, CommandStart(cmd));
    return MZ_RESULT_SUCCESS; 
}

mzResult MZAPI_CALL WaitEvent(uint64_t eventHandle)
{
	((std::future<void>*)eventHandle)->wait();
	delete ((std::future<void>*)eventHandle);
	return MZ_RESULT_SUCCESS;
}

// Recordable calls
mzResult MZAPI_CALL Copy(mzCmd inCmd,
						 const mzResourceShareInfo* src,
						 const mzResourceShareInfo* dst,
						 const char* benchmark)
{
	if (!inCmd)
	{
		ENQUEUE_RENDER_COMMAND(Copy, [src = *src, dst = *dst, benchmark](auto cmd) {
			Copy(cmd.get(), &src, &dst, benchmark);
		});
		return MZ_RESULT_SUCCESS;
	}
	auto cmd = CommandStart(inCmd);

	constexpr u32 MAX = (u32)MZ_RESOURCE_TYPE_TEXTURE;
	constexpr u32 BUF = (u32)MZ_RESOURCE_TYPE_BUFFER;
	constexpr u32 IMG = (u32)MZ_RESOURCE_TYPE_TEXTURE;

	auto RFN = [src, dst](auto cmd) {
		switch ((u32)src->Info.Type | ((u32)dst->Info.Type << MAX))
		{
		default: return MZ_RESULT_INVALID_ARGUMENT;
		case BUF | (BUF << MAX):
			ImportBufferDef(GVkDevice, dst)->Upload(cmd, ImportBufferDef(GVkDevice, src));
			break;
		case BUF | (IMG << MAX):
			ImportImageDef(GVkDevice, dst)->Upload(cmd, ImportBufferDef(GVkDevice, src));
			break;
		case IMG | (BUF << MAX):
			ImportImageDef(GVkDevice, src)->Download(cmd, ImportBufferDef(GVkDevice, dst));
			break;
		case IMG | (IMG << MAX):
			auto dstImg = ImportImageDef(GVkDevice, dst);
			auto srcImg = ImportImageDef(GVkDevice, src);
			if (srcImg->GetEffectiveExtent() == dstImg->GetEffectiveExtent() &&
				srcImg->GetEffectiveFormat() == dstImg->GetEffectiveFormat())
			{
				dstImg->CopyFrom(cmd, srcImg);
			}
			else
			{
				dstImg->BlitFrom(cmd, srcImg, (VkFilter)dst->Info.Texture.Filter);
			}
			break;
		}
	};

	if (benchmark)
	{
		u32 frames = 30;
		std::string bench_key = benchmark;
		if (auto p = bench_key.find_last_of(':'); std::string::npos != p)
		{
			frames = atoi(bench_key.c_str() + p + 1);
			bench_key.erase(bench_key.begin() + p, bench_key.end());
		}

		if (auto re = GVkDevice->GetQPool()->PerfScope(frames, bench_key, cmd, std::move(RFN)))
		{
			auto t = *re;
			mzEngine.LogI("%s took: %u ns (%u us) (%u ms)", t.count(), std::chrono::duration_cast<std::chrono::microseconds>(t).count(), std::chrono::duration_cast<std::chrono::milliseconds>(t).count());
		}
	}
	else
	{
		RFN(cmd);
	}

	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

}



