#include <Nodos/PluginAPI.h>
#include <Builtins_generated.h>
#include <Nodos/PluginHelpers.hpp>
#include <AppService_generated.h>
#include <AppEvents_generated.h>
#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include "Names.h"

NOS_INIT();
nosResult RegisterWebRTCPlayer(nosNodeFunctions* outFunctions);
nosResult RegisterWebRTCStreamer(nosNodeFunctions* outFunctions);
void RegisterWebRTCSignalingServer(nosNodeFunctions* outFunctions);

nosVulkanSubsystem* nosVulkan = nullptr;

NOS_REGISTER_NAME(In)
NOS_REGISTER_NAME(Out)
NOS_REGISTER_NAME(ServerIP)
NOS_REGISTER_NAME(StreamerID)
NOS_REGISTER_NAME(MaxFPS)
NOS_REGISTER_NAME(WebRTCPlayer);
NOS_REGISTER_NAME(WebRTCStreamer);
NOS_REGISTER_NAME(TargetBitrate);

NOS_REGISTER_NAME(RGBtoYUV420_Compute_Shader);
NOS_REGISTER_NAME(RGBtoYUV420_Compute_Pass);
NOS_REGISTER_NAME(YUV420toRGB_Compute_Shader);
NOS_REGISTER_NAME(YUV420toRGB_Compute_Pass);
NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(PlaneY);
NOS_REGISTER_NAME(PlaneU);
NOS_REGISTER_NAME(PlaneV);

extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions) {
		*outCount = (size_t)(3);
		if (!outFunctions)
			return NOS_RESULT_SUCCESS;

		nosResult returnRes = nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), NOS_VULKAN_SUBSYSTEM_VERSION_MAJOR, 0, (void**)&nosVulkan);
		if (returnRes != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;
		 
		nosResult res = RegisterWebRTCStreamer(outFunctions[0]);
		if (res != NOS_RESULT_SUCCESS)
			returnRes = res;
		res = RegisterWebRTCPlayer(outFunctions[1]);
		if (res != NOS_RESULT_SUCCESS)
			returnRes = res;
		RegisterWebRTCSignalingServer(outFunctions[2]);

		return NOS_RESULT_SUCCESS;
	}
}