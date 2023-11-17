// Copyright MediaZ AS. All Rights Reserved.
#include <mzVulkanSubsystem/mzVulkanSubsystem.h>
#include <MediaZ/SubsystemAPI.h>
#include <mzVulkan/Common.h>

#include "Services.h"

MZ_INIT();

namespace mz::vk
{
void Bind(mzVulkanSubsystem& subsystem)
{
	subsystem.Begin = vkss::Begin;
	subsystem.End = vkss::End;
	subsystem.WaitEvent = vkss::WaitEvent;
}
}

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportSubsystem(void** subsystemContext)
{
	mz::vkss::Initialize();
	auto subsystem = new mzVulkanSubsystem;
	mz::vk::Bind(*subsystem);
	return MZ_RESULT_SUCCESS;
}

MZAPI_ATTR bool MZAPI_CALL mzUnloadSubsystem(void* subsystemContext)
{
	auto ret = mz::vkss::Deinitialize();
	delete static_cast<mzVulkanSubsystem*>(subsystemContext);
	return ret == MZ_RESULT_SUCCESS;
}

}
