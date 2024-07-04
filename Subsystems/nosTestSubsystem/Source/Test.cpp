// Copyright MediaZ Teknoloji A.S. All Rights Reserved.
#include <nosTestSubsystem/TestSubsystem.h>
#include <Nodos/SubsystemAPI.h>
#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <Nodos/Helpers.hpp>
NOS_INIT_WITH_MIN_REQUIRED_MINOR(7); // APITransition
NOS_VULKAN_INIT();

void OnRequestedSubsystemUnloaded(nosName name, int versionMajor, int versionMinor)
{
	if (name == NOS_NAME(NOS_VULKAN_SUBSYSTEM_NAME))
		nosVulkan = nullptr;	
}

extern "C"
{


NOSAPI_ATTR nosResult NOSAPI_CALL OnRequest(uint32_t minor, void** outSubsystemContext)
{
	static TestSubsystem testSubsystem = {};
	*outSubsystemContext = &testSubsystem;
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
{
	subsystemFunctions->OnRequest = OnRequest;
	subsystemFunctions->OnRequestedSubsystemUnloaded = OnRequestedSubsystemUnloaded;
	auto ret = RequestVulkanSubsystem();
	if (ret != NOS_RESULT_SUCCESS)
		return ret;
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR nosResult NOSAPI_CALL nosUnloadSubsystem()
{
	return NOS_RESULT_SUCCESS;
}

}
