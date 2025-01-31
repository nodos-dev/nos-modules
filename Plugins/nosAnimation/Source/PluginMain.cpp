// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

// Includes
#include <Nodos/PluginAPI.h>
#include <Nodos/PluginHelpers.hpp>

// Subsystem dependencies
#include <nosVulkanSubsystem/nosVulkanSubsystem.h>

NOS_INIT()
NOS_VULKAN_INIT()

NOS_BEGIN_IMPORT_DEPS()
	NOS_VULKAN_IMPORT()
NOS_END_IMPORT_DEPS()

namespace nos::animation
{
enum Nodes : int
{
    Animate = 0,
    Count
};

nosResult RegisterAnimate(nosNodeFunctions*);

nosResult NOSAPI_CALL ExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	if (!outList)
	{
		*outSize = Nodes::Count;
		return NOS_RESULT_SUCCESS;
	}

#define GEN_CASE_NODE(name)					\
	case Nodes::name: {					\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (int i = 0; i < Nodes::Count; ++i)
	{
		auto node = outList[i];
		switch ((Nodes)i) {
		default:
			break;
			GEN_CASE_NODE(Animate)
		}
	}
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportPlugin(nosPluginFunctions* outFunctions)
{
	outFunctions->ExportNodeFunctions = ExportNodeFunctions;
	return NOS_RESULT_SUCCESS;
}
}
} // namespace nos::noise