#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include "NVVFX_Names.h"

NOS_INIT();

NOS_REGISTER_NAME(NVVFX_AR);
NOS_REGISTER_NAME(NVVFX_SuperRes);
NOS_REGISTER_NAME(NVVFX_AIGreenScreen);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(UpscaleFactor);
NOS_REGISTER_NAME(ModelsPath);
NOS_REGISTER_NAME(UpscaleStrength);
NOS_REGISTER_NAME(Out);

void RegisterNVVFX_AR(nosNodeFunctions* outFunctions);
void RegisterNVVFX_SuperRes(nosNodeFunctions* outFunctions);
void RegisterNVVFX_AIGreenScreen(nosNodeFunctions* outFunctions);

extern nosVulkanSubsystem* nosVulkan = nullptr;

extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions) {
		*outCount = (size_t)(3);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		nosResult returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 1, 0, (void**)&nosVulkan);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		RegisterNVVFX_AR(outFunctions[0]);
		RegisterNVVFX_SuperRes(outFunctions[1]);
		RegisterNVVFX_AIGreenScreen(outFunctions[2]);

		return NOS_RESULT_SUCCESS;
	}
}