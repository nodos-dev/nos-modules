#include <Nodos/PluginHelpers.hpp>
#include <Nodos/Helpers.hpp>

NOS_INIT_WITH_MIN_REQUIRED_MINOR(13) // Remember to remove this when transitioning to the next major version of the Nodos Plugin SDK.

namespace nos::reflect
{

enum Nodes : size_t
{	// CPU nodes
	Make = 0,
	MakeDynamic,
	Break,
	Indexer,
	Array,
	Delay,
	Arithmetic,
	Count
};

nosResult RegisterMake(nosNodeFunctions* node);
nosResult RegisterBreak(nosNodeFunctions* node);
nosResult RegisterIndexer(nosNodeFunctions* node);
nosResult RegisterArray(nosNodeFunctions* node);
nosResult RegisterDelay(nosNodeFunctions* node);
nosResult RegisterArithmetic(nosNodeFunctions* node);

void OnVulkanSubsystemUnloaded();

NOS_REGISTER_NAME_SPACED(VulkanSubsystemName, "nos.sys.vulkan")
void NOSAPI_CALL OnRequestedSubsystemUnloaded(nosName subsystemName, int versionMajor, int versionMinor)
{
	if (subsystemName == NSN_VulkanSubsystemName)
		OnVulkanSubsystemUnloaded();		
}

nosResult NOSAPI_CALL ExportNodeFunctions(size_t* outCount, nosNodeFunctions** outFunctions)
{
	*outCount = (size_t)(Nodes::Count);
	if (!outFunctions)
		return NOS_RESULT_SUCCESS;
	
#define GEN_CASE_NODE(name)					\
	case Nodes::name: {						\
		auto ret = Register##name(node);	\
		if (NOS_RESULT_SUCCESS != ret)		\
			return ret;						\
		break;								\
	}

	for (size_t i = 0; i < (size_t)Nodes::Count; ++i)
	{
		auto node = outFunctions[i];
		switch ((Nodes)i)
		{
			GEN_CASE_NODE(Make)
			GEN_CASE_NODE(Break)
			GEN_CASE_NODE(Indexer)
			GEN_CASE_NODE(Array)
			GEN_CASE_NODE(Delay)
			GEN_CASE_NODE(Arithmetic)
		}
	}

#undef GEN_CASE_NODE
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
/// Nodos calls this function to initialize the plugin & retrieve the plugin's functions.
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportPlugin(nosPluginFunctions* outPluginFunctions)
{
	outPluginFunctions->ExportNodeFunctions = ExportNodeFunctions;
	outPluginFunctions->OnRequestedSubsystemUnloaded = OnRequestedSubsystemUnloaded;
	return NOS_RESULT_SUCCESS;
}
}

}