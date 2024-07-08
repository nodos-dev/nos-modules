// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginAPI.h>

#include <Builtins_generated.h>

#include <Nodos/PluginHelpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVulkanSubsystem/Helpers.hpp>

#include "nosUtil/Stopwatch.hpp"

#include "WebcamStream.h"

NOS_INIT_WITH_MIN_REQUIRED_MINOR(0)
NOS_VULKAN_INIT();

namespace nos::webcam
{
enum WebcamNodes : int
{	// CPU nodes
	WebcamReader = 0,
	WebcamStream,
	Count
};

nosResult RegisterWebcamReader(nosNodeFunctions* function);
nosResult RegisterWebcamStream(nosNodeFunctions* function);

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
{
	*outCount = (size_t)(WebcamNodes::Count);
	if (!outFunctions)
		return NOS_RESULT_SUCCESS;

	NOS_RETURN_ON_FAILURE(RequestVulkanSubsystem());
		
	WebcamStreamManager::Start();

#define GEN_CASE_NODE(name)				\
	case WebcamNodes::name: {					\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (int i = 0; i < WebcamNodes::Count; ++i)
	{
		auto node = outFunctions[i];
		switch ((WebcamNodes)i) {
			default:
				break;
			GEN_CASE_NODE(WebcamReader)
			GEN_CASE_NODE(WebcamStream)
		}
	}
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR void NOSAPI_CALL nosUnloadPlugin()
{
	WebcamStreamManager::Stop();
}

}
} // namespace nos::test
