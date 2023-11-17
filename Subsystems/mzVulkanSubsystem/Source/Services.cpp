#include "Services.h"

#include "MediaZ/SubsystemAPI.h"

#include "mzVulkan/Device.h"
#include "mzVulkan/Command.h"

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
	//RenderThread::Create(); //TODO
	//Use<RenderThread>()->Start(); //TODO

    return MZ_RESULT_SUCCESS;
}

mzResult Deinitialize()
{
	//Use<RenderThread>()->Stop(); //TODO
	//RenderThread::Destroy(); //TODO
	GVkDevice = nullptr;
	GVkCtx = nullptr;
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


}



