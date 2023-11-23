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
#include <mzVulkan/Renderpass.h>

// SDK
#include <MediaZ/Name.hpp>
#include <mzFlatBuffersCommon.h>

// std
#include <future>
#include <chrono>

// TODO:
// 1. Combine benchmark codes into one function to avoid duplicate code.

namespace mz::vkss
{
#include "CAPIStructHelpers.inl"

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

void Bind(mzVulkanSubsystem& subsystem)
{
	subsystem.Begin = vkss::Begin;
	subsystem.End = vkss::End;
	subsystem.WaitEvent = vkss::WaitEvent;
	subsystem.Copy = vkss::Copy;
	subsystem.RunPass = vkss::RunPass;
	subsystem.RunPass2 = vkss::RunPass2;
	subsystem.RunComputePass = vkss::RunComputePass;
	subsystem.Clear = vkss::ClearTexture;
	subsystem.Download = vkss::DownloadTexture;
	subsystem.Create = vkss::CreateResource;
	subsystem.Destroy = vkss::DestroyResource;
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

void ExportBufferDef(rc<vk::Buffer> buf, mzResourceShareInfo* info)
{
	info->Memory.Handle = (u64)buf.get();
	info->Memory.PID = mz::vk::PlatformGetCurrentProcessId();
	info->Memory.Memory = (u64)buf->Allocation.OsHandle;
	info->Memory.Offset = buf->Allocation.GetOffset();
	info->Memory.Type = (u32)buf->Allocation.ExternalMemoryHandleTypes;
	info->Info.Type = MZ_RESOURCE_TYPE_BUFFER;
	info->Info.Buffer.Size = buf->Allocation.GetSize();
	info->Info.Buffer.Usage = (mzBufferUsage)buf->Usage;
}

void ExportImageDef(rc<vk::Image> image, mzResourceShareInfo* tex)
{
	mz::vk::MemoryExportInfo Ext = image->GetExportInfo();
	tex->Memory.Handle = (u64)image.get();
	tex->Memory.PID = Ext.PID;
	tex->Memory.Memory = (u64)Ext.Memory;
	tex->Memory.Offset = Ext.Offset;
	tex->Memory.Type = (u32)Ext.Type;
	tex->Info.Type = MZ_RESOURCE_TYPE_TEXTURE;
	tex->Info.Texture.Width = image->GetExtent().width;
	tex->Info.Texture.Height = image->GetExtent().height;
	tex->Info.Texture.Format = (mzFormat)image->GetFormat();
	tex->Info.Texture.Usage = (mzImageUsage)image->Usage;
	tex->Info.Texture.Semaphore = (u64)(image->ExtSemaphore ? image->ExtSemaphore->Handle : 0);
	tex->Info.Texture.FieldType = MZ_TEXTURE_FIELD_TYPE_PROGRESSIVE;
}

void UpdateTextureSizePresetFromSize(mz::fb::TTexture* tex)
{
	if (tex->width == 1920 && tex->height == 1080)
		tex->size = mz::fb::SizePreset::HD;
	else if (tex->width == 3840 && tex->height == 2160)
		tex->size = mz::fb::SizePreset::ULTRA_HD;
	else
		tex->size = mz::fb::SizePreset::CUSTOM;
}

void FillDefaultParameters(mzTextureInfo* texture)
{
	mzBuffer defaultData{};
	mzEngine.GetDefaultValueOfType(MZ_NAME_STATIC("mz.fb.Texture"), &defaultData);
	if (!defaultData.Data)
	{
		mzEngine.LogE("Failed to get default data for mz.fb.Texture");
		return;
	}
	auto tex = flatbuffers::GetRoot<mz::fb::Texture>(defaultData.Data);
	texture->Width = texture->Width ? texture->Width : tex->width();
	texture->Height = texture->Height ? texture->Height : tex->height();
	texture->Format = (u32)texture->Format ? texture->Format : (mzFormat)tex->format();
	texture->Usage = (u32)texture->Usage ? texture->Usage : (mzImageUsage)tex->usage();
}

void FillDefaultParameters(mzBufferInfo* data)
{
	mzBuffer defaultData{};
	mzEngine.GetDefaultValueOfType(MZ_NAME_STATIC("mz.fb.Buffer"), &defaultData);
	if (!defaultData.Data)
	{
		mzEngine.LogE("Failed to get default data for mz.fb.Buffer");
		return;
	}
	auto defaultBuf = flatbuffers::GetRoot<mz::fb::Buffer>(defaultData.Data);
	data->Size = data->Size ? data->Size : defaultBuf->size();
	data->Usage = (u32)data->Usage ? (mzBufferUsage)data->Usage : (mzBufferUsage)defaultBuf->usage();
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
			mzEngine.LogI("Benchmark: %s took %u ns (%u us) (%u ms)", bench_key.c_str(), t.count(), std::chrono::duration_cast<std::chrono::microseconds>(t).count(), std::chrono::duration_cast<std::chrono::milliseconds>(t).count());
		}
	}
	else
	{
		RFN(cmd);
	}

	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

auto ConvertVertices(mzVertexData v, vk::VertexData* verts)
{
	if (!verts || !v.Buffer.Memory.Handle)
		return (vk::VertexData*)0;

	*verts = vk::VertexData{
		.Buffer = ImportBufferDef(GVkDevice, &v.Buffer),
		.VertexOffset = v.VertexOffset,
		.IndexOffset = v.IndexOffset,
		.NumIndices = v.IndexCount,
		.DepthWrite = (bool)v.DepthWrite,
		.DepthTest = (bool)v.DepthTest,
		.DepthFunc = (VkCompareOp)v.DepthFunc,
	};
	return verts;
}

void BindToPass(rc<vk::Basepass> RP, const mzShaderBinding* binding)
{
	auto name = mz::Name(binding->Name).AsString();
	switch (RP->GetUniformClass(name))
	{
	case vk::Basepass::IMAGE_ARRAY:
		// TODO
		{
			std::vector<rc<vk::Image>> images;
			for (u32 i = 0; i < binding->Size / sizeof(mzResourceShareInfo); ++i)
				images.push_back(ImportImageDef(GVkDevice, &binding->Resource[i]));
			RP->BindResource(name, images, (VkFilter)binding->Resource->Info.Texture.Filter);
		}
		break;
	case vk::Basepass::IMAGE:
		RP->BindResource(name,
						 ImportImageDef(GVkDevice, binding->Resource),
						 (VkFilter)binding->Resource->Info.Texture.Filter);
		break;
	case vk::Basepass::BUFFER: RP->BindResource(name, ImportBufferDef(GVkDevice, binding->Resource)); break;
	case vk::Basepass::UNIFORM: RP->BindData(name, binding->Data, binding->Size); break;
	}
}

mzResult MZAPI_CALL RunPass(mzCmd inCmd, const mzRunPassParams* params)
{
	if (!inCmd)
	{
		ENQUEUE_RENDER_COMMAND(RunPass, [copiedParams = OwnedRunPassParams(*params)](rc<vk::CommandBuffer> cmd) {
			RunPass(cmd.get(), &copiedParams);
		});
		return MZ_RESULT_SUCCESS;
	}
	auto cmd = CommandStart(inCmd);

	auto RP = GVkDevice->GetGlobal<rc<vk::Renderpass>>(mz::Name(params->Key).AsCStr());
	RP->Lock();
	for (u32 i = 0; i < params->BindingCount; ++i)
		BindToPass(RP, &params->Bindings[i]);

	vk::VertexData verts;
	RP->Exec(cmd,
			 ImportImageDef(GVkDevice, &params->Output),
			 ConvertVertices(params->Vertices, &verts),
			 !params->DoNotClear,
			 0,
			 0.0f,
			 {params->ClearCol.x, params->ClearCol.y, params->ClearCol.z, params->ClearCol.w});
	RP->Unlock();
	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

void PrepassTransition(rc<vk::Basepass> RP, rc<vk::CommandBuffer> cmd, const mzShaderBinding* binding)
{
	auto name = mz::Name(binding->Name).AsString();
	switch (RP->GetUniformClass(name))
	{
	case vk::Basepass::IMAGE_ARRAY:
		// TODO
		assert(0);
	case vk::Basepass::IMAGE:
		RP->TransitionInput(cmd, name, ImportImageDef(GVkDevice, binding->Resource));
		break;
	}
}

mzResult MZAPI_CALL RunPass2(mzCmd inCmd, const mzRunPass2Params* params)
{
	if (!inCmd)
	{
		ENQUEUE_RENDER_COMMAND(RunPass2, [params = OwnedRunPass2Params(*params)](rc<vk::CommandBuffer> cmd) {
			RunPass2(cmd.get(), &params);
		});
		return MZ_RESULT_SUCCESS;
	}
	auto RP = GVkDevice->GetGlobal<rc<vk::Renderpass>>(mz::Name(params->Key).AsCStr());

	if (!RP)
		return MZ_RESULT_INVALID_ARGUMENT;

	assert(RP->PL->MainShader->Stage == VK_SHADER_STAGE_FRAGMENT_BIT);

	auto RFN = [params, &RP](auto cmd) {
		RP->Lock();
		for (u32 i = 0; i < params->DrawCallCount; ++i)
			for (u32 j = 0; j < params->DrawCalls[i].BindingCount; ++j)
				PrepassTransition(RP, cmd, &params->DrawCalls[i].Bindings[j]);

		auto output = ImportImageDef(GVkDevice, &params->Output);
		RP->Begin(cmd,
				  output,
				  params->Wireframe,
				  !params->DoNotClear,
				  0,
				  0.0f,
				  {params->ClearCol.x, params->ClearCol.y, params->ClearCol.z, params->ClearCol.w});

		for (u32 i = 0; i < params->DrawCallCount; ++i)
		{
			for (u32 j = 0; j < params->DrawCalls[i].BindingCount; ++j)
				BindToPass(RP, &params->DrawCalls[i].Bindings[j]);

			RP->BindResources(cmd);
			RP->DescriptorSets.clear();
			vk::VertexData verts;
			RP->Draw(cmd, ConvertVertices(params->DrawCalls[i].Vertices, &verts));
		}

		RP->End(cmd);
		RP->Unlock();
	};

	auto cmd = CommandStart(inCmd);
	if (params->Benchmark)
	{
		if (auto re = GVkDevice->GetQPool()->PerfScope(params->Benchmark, mz::Name(params->Key).AsCStr(), cmd, std::move(RFN)))
		{
			auto t = *re;
			mzEngine.LogI("Benchmark: %s took %u ns (%u us) (%u ms)",
						  mzEngine.GetString(params->Key),
						  t.count(),
						  std::chrono::duration_cast<std::chrono::microseconds>(t).count(),
						  std::chrono::duration_cast<std::chrono::milliseconds>(t).count());
		}
	}
	else
	{
		RFN(cmd);
	}
	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

mzResult MZAPI_CALL RunComputePass(mzCmd inCmd, const mzRunComputePassParams* params) 
{
    if (!inCmd)
    {
		ENQUEUE_RENDER_COMMAND(RunComputePass, [params = OwnedRunComputePassParams(*params)](rc<vk::CommandBuffer> cmd) {
			RunComputePass(cmd.get(), &params);
		});
		return MZ_RESULT_SUCCESS;
    }
    auto CP = GVkDevice->GetGlobal<rc<vk::Computepass>>(mz::Name(params->Key).AsCStr());
    assert(CP->PL->MainShader->Stage == VK_SHADER_STAGE_COMPUTE_BIT);
	CP->Lock();
	auto cmd = CommandStart(inCmd);

    for(u32 i = 0; i < params->BindingCount; ++i)
    {
        PrepassTransition(CP, cmd, &params->Bindings[i]);
        BindToPass(CP, &params->Bindings[i]);
    }

    CP->BindResources(cmd);

    if(params->Benchmark)
    {
        if(auto re = GVkDevice->GetQPool()->PerfScope(params->Benchmark, mz::Name(params->Key).AsCStr(), cmd, [&CP, ds = params->DispatchSize](auto cmd) {
            CP->Dispatch(cmd, ds.x, ds.y);
        }))
        {
            auto t = *re;
			mzEngine.LogI("Benchmark: Compute Pass (%d, %d) %s took %u ns (%u us) (%u ms)", params->DispatchSize.x, params->DispatchSize.y,
						  mzEngine.GetString(params->Key),
						  t.count(),
						  std::chrono::duration_cast<std::chrono::microseconds>(t).count(),
						  std::chrono::duration_cast<std::chrono::milliseconds>(t).count());
        }
    }
    else 
    {
        CP->Dispatch(cmd, params->DispatchSize.x, params->DispatchSize.y);
    }
    CommandFinish(inCmd, cmd);
	CP->Bindings.clear();
    CP->Unlock();
	return MZ_RESULT_SUCCESS;
}

mzResult MZAPI_CALL ClearTexture(mzCmd inCmd, const mzResourceShareInfo* texture, mzVec4 color)
{
	if (!inCmd)
	{
		ENQUEUE_RENDER_COMMAND(ClearTexture, [texture = *texture, color](rc<vk::CommandBuffer> cmd) {
			ClearTexture(cmd.get(), &texture, color);
		});
		return MZ_RESULT_SUCCESS;
	}

	if (MZ_RESOURCE_TYPE_TEXTURE != texture->Info.Type)
	{
		return MZ_RESULT_INVALID_ARGUMENT;
	}
	auto cmd = CommandStart(inCmd);
	ImportImageDef(GVkDevice, texture)
		->Clear(cmd,
				VkClearColorValue{.float32{
					color.x,
					color.y,
					color.z,
					color.w,
				}});
	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

mzResult MZAPI_CALL DownloadTexture(mzCmd inCmd, const mzResourceShareInfo* texture, mzResourceShareInfo* outBuffer)
{
	if (!outBuffer || MZ_RESOURCE_TYPE_TEXTURE != texture->Info.Type)
	{
		return MZ_RESULT_INVALID_ARGUMENT;
	}
	auto img = ImportImageDef(GVkDevice, texture);
	rc<vk::Buffer> stagingBuffer = GResources->Create<vk::Buffer>({
		.Size = (u32)img->Allocation.GetSize(),
		.Usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	});
	auto cmd = CommandStart(inCmd);
	img->Download(cmd, stagingBuffer);
	ExportBufferDef(stagingBuffer, outBuffer);
	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

mzResult MZAPI_CALL CreateResource(mzResourceShareInfo* res)
{
	Use<RenderThread>()->Flush();
	if (res->Memory.Handle)
		mzEngine.LogW("CreateResource was called on an existing resource! Make sure you don't leak your resources!");
	switch (res->Info.Type)
	{
	default: UNREACHABLE;
	case MZ_RESOURCE_TYPE_BUFFER: {
		auto buf = res->Info.Buffer;
		FillDefaultParameters(&buf);
		auto vkBuf = GResources->Create<vk::Buffer>({
			.Size = (u32)buf.Size,
			.Mapped = !bool(buf.Usage & MZ_BUFFER_USAGE_NOT_HOST_VISIBLE),
			.VRAM = bool(buf.Usage & MZ_BUFFER_USAGE_DEVICE_MEMORY),
			.Usage = VkBufferUsageFlags((u32)buf.Usage &
										// these bits mean something else to vulkan
										// so mask them out to make sure to not create a buffer with irrelevant flags
										~(MZ_BUFFER_USAGE_NOT_HOST_VISIBLE | MZ_BUFFER_USAGE_DEVICE_MEMORY)),
		});
		ExportBufferDef(vkBuf, res);
		break;
	}
	case MZ_RESOURCE_TYPE_TEXTURE: {
		auto img = res->Info.Texture;

		if (!vk::IsFormatSupportedByDevice(VkFormat(img.Format), GVkDevice->PhysicalDevice))
		{
			mzEngine.LogE("Image create error! Plugin requested an unsupported texture type by device");
			return MZ_RESULT_FAILED;
		}

		FillDefaultParameters(&img);
		auto vkImg = GResources->Create<vk::Image>({
			.Extent = {.width = img.Width, .height = img.Height},
			.Format = VkFormat(img.Format),
			.Usage = VkImageUsageFlags(img.Usage),
			// .Filtering = (VkFilter)req->filtering,
			// .Type = VkExternalMemoryHandleTypeFlagBits(req->type),
		});
		ExportImageDef(vkImg, res);
		break;
	}
	}

	return MZ_RESULT_SUCCESS;
}

// Handles are written, other info is used as creation info.
mzResult MZAPI_CALL DestroyResource(const mzResourceShareInfo* resource)
{
	Use<RenderThread>()->Flush();
	switch (resource->Info.Type)
	{
	case MZ_RESOURCE_TYPE_BUFFER: GResources->Destroy<vk::Buffer>(resource->Memory.Handle); break;
	case MZ_RESOURCE_TYPE_TEXTURE: GResources->Destroy<vk::Image>(resource->Memory.Handle); break;
	default: return MZ_RESULT_FAILED;
	}
	return MZ_RESULT_SUCCESS;
}


}



