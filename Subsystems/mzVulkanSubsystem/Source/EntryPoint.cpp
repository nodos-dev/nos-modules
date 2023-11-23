// Copyright MediaZ AS. All Rights Reserved.
#include <mzVulkanSubsystem/mzVulkanSubsystem.h>
#include <MediaZ/SubsystemAPI.h>
#include <mzVulkan/Common.h>

MZ_INIT();

#include "Services.h"

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportSubsystem(void** subsystemContext)
{
	mz::vkss::Initialize();
	auto subsystem = new mzVulkanSubsystem;
	mz::vkss::Bind(*subsystem);
	return MZ_RESULT_SUCCESS;
}

MZAPI_ATTR bool MZAPI_CALL mzUnloadSubsystem(void* subsystemContext)
{
	auto ret = mz::vkss::Deinitialize();
	delete static_cast<mzVulkanSubsystem*>(subsystemContext);
	return ret == MZ_RESULT_SUCCESS;
}

}
