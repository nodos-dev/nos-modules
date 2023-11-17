// Copyright MediaZ AS. All Rights Reserved.
#include <mzVulkanSubsystem/mzVulkanSubsystem.h>
#include <MediaZ/SubsystemAPI.h>
#include <mzVulkan/Common.h>

MZ_INIT();

namespace mz::vk
{

// Recordable calls
mzResult MZAPI_CALL Copy(mzCmd inCmd, const mzResourceShareInfo* src, const mzResourceShareInfo* dst, const char* benchmark)
{
	if (!inCmd)
	{
		ENQUEUE_RENDER_COMMAND(Copy, [src = *src, dst = *dst, benchmark](rc<vk::CommandBuffer> cmd) {
			Copy(cmd.get(), &src, &dst, benchmark);
		});
		return MZ_RESULT_SUCCESS;
	}
	auto cmd = CommandStart(inCmd);

	constexpr u32 MAX = (u32)app::Resource::MAX;
	constexpr u32 BUF = (u32)app::Resource::mz_fb_Buffer;
	constexpr u32 IMG = (u32)app::Resource::mz_fb_Texture;

	auto RFN = [src, dst](auto cmd) {
		switch ((u32)src->Info.Type | ((u32)dst->Info.Type << MAX))
		{
		default: UNREACHABLE;
		case BUF | (BUF << MAX):
			ImportBufferDef(GVkDevice.load(), dst)->Upload(cmd, ImportBufferDef(GVkDevice.load(), src));
			break;
		case BUF | (IMG << MAX):
			ImportImageDef(GVkDevice.load(), dst)->Upload(cmd, ImportBufferDef(GVkDevice.load(), src));
			break;
		case IMG | (BUF << MAX):
			ImportImageDef(GVkDevice.load(), src)->Download(cmd, ImportBufferDef(GVkDevice.load(), dst));
			break;
		case IMG | (IMG << MAX):
			auto dstImg = ImportImageDef(GVkDevice.load(), dst);
			auto srcImg = ImportImageDef(GVkDevice.load(), src);
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

		if (auto re = GVkDevice.load()->GetQPool()->PerfScope(frames, bench_key, cmd, std::move(RFN)))
		{
			auto t = *re;
			mz::li() << bench_key << " took: " << t << " (" << std::chrono::duration_cast<Micro>(t) << ")"
					 << " (" << std::chrono::duration_cast<Milli>(t) << ")"
					 << "\n";
		}
	}
	else
	{
		RFN(cmd);
	}

	CommandFinish(inCmd, cmd);
	return MZ_RESULT_SUCCESS;
}

void Bind(mzVulkanSubsystem& subsystem)
{
	subsystem.Copy = Copy;
}
}

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportSubsystem(void** subsystemContext)
{
	auto subsystem = new mzVulkanSubsystem;
	mz::vk::Bind(*subsystem);
	return MZ_RESULT_SUCCESS;
}

MZAPI_ATTR bool MZAPI_CALL mzUnloadSubsystem(void* subsystemContext)
{
	delete static_cast<mzVulkanSubsystem*>(subsystemContext);
	return true;
}

}
