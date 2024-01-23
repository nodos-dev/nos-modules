// Copyright Nodos AS. All Rights Reserved.

// Includes
#include <Nodos/PluginHelpers.hpp>
#include <glm/glm.hpp>
#include <Builtins_generated.h>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>

NOS_INIT();

NOS_REGISTER_NAME(Input);
NOS_REGISTER_NAME(Output);
NOS_REGISTER_NAME(In);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME(Path);
NOS_REGISTER_NAME(sRGB);

namespace nos::utilities
{

nosVulkanSubsystem* nosVulkan = nullptr;

enum Utilities : int
{	// CPU nodes
	Resize = 0,
	ChannelViewer,
	Merge,
	Time,
	ReadImage,
	WriteImage,
	Interlace,
	Deinterlace,
	Count
};

nosResult RegisterMerge(nosNodeFunctions*);
nosResult RegisterTime(nosNodeFunctions*);
nosResult RegisterReadImage(nosNodeFunctions*);
nosResult RegisterWriteImage(nosNodeFunctions*);
nosResult RegisterChannelViewer(nosNodeFunctions*);
nosResult RegisterResize(nosNodeFunctions*);
nosResult RegisterInterlace(nosNodeFunctions*);
nosResult RegisterDeinterlace(nosNodeFunctions*);

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
    *outSize = Utilities::Count;
	if (!outList)
		return NOS_RESULT_SUCCESS;

	auto ret = nosEngine.RequestSubsystem(NOS_NAME_STATIC("nos.sys.vulkan"), NOS_VULKAN_SUBSYSTEM_VERSION_MAJOR, 0, (void**)&nosVulkan);
	if (ret != NOS_RESULT_SUCCESS)
		return ret;

#define GEN_CASE_CPU_NODE(name)				\
	case Utilities::name: {					\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (int i = 0; i < Utilities::Count; ++i)
	{
		auto node = outList[i];
		switch ((Utilities)i) {
			default:
				break;
			GEN_CASE_CPU_NODE(Merge)
			GEN_CASE_CPU_NODE(Time)
			GEN_CASE_CPU_NODE(ReadImage)
			GEN_CASE_CPU_NODE(WriteImage)
			GEN_CASE_CPU_NODE(ChannelViewer)
			GEN_CASE_CPU_NODE(Resize)
			GEN_CASE_CPU_NODE(Interlace)
			GEN_CASE_CPU_NODE(Deinterlace)
		}
	}
	return NOS_RESULT_SUCCESS;
}
}
}
