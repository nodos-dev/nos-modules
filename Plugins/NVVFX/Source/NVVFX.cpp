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
char* g_nvVFXSDKPath = NULL;
char* g_nvCVImagePath = NULL;

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

		nosSubsystemContext deps = {};
		returnRes = nosEngine.RequestSubsystem2(NOS_NAME_STATIC("nos.NVVFX.Dependencies"), 1, 0, &deps);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		char* sdkPathDyn = new char[260];
		memset(sdkPathDyn, 0, 260);
		std::string sdkPathStr = std::string(deps.Context.RootFolderPath) + "\\NVVFX_SDK";
		const char* sdkPath = sdkPathStr.c_str();
		memcpy(sdkPathDyn, sdkPath, strlen(sdkPath));
		g_nvVFXSDKPath = sdkPathDyn;
		g_nvCVImagePath = sdkPathDyn;
		
		NSN_ModelsPath = nos::Name(std::string(std::string(deps.Context.RootFolderPath) + "\\NVVFX_SDK\\models").c_str());
		
		RegisterNVVFX_AR(outFunctions[0]);
		RegisterNVVFX_SuperRes(outFunctions[1]);
		RegisterNVVFX_AIGreenScreen(outFunctions[2]);

		return NOS_RESULT_SUCCESS;
	}
}