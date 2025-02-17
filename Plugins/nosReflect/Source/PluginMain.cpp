#include <Nodos/PluginHelpers.hpp>
#include <Nodos/Helpers.hpp>

#include <nosVulkanSubsystem/nosVulkanSubsystem.h>
#include <nosVariableSubsystem/nosVariableSubsystem.h>

NOS_INIT()
NOS_VULKAN_INIT()
NOS_SYS_VARIABLES_INIT()

NOS_BEGIN_IMPORT_DEPS()
	NOS_VULKAN_IMPORT()
	NOS_SYS_VARIABLES_IMPORT()
NOS_END_IMPORT_DEPS()

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
	ScalarArithmetic,
	IndexOf,
	IsEqual,
	SetVariable,
	GetVariable,
	Count
};

nosResult RegisterMake(nosNodeFunctions* node);
nosResult RegisterMakeDynamic(nosNodeFunctions* node);
nosResult RegisterBreak(nosNodeFunctions* node);
nosResult RegisterIndexer(nosNodeFunctions* node);
nosResult RegisterArray(nosNodeFunctions* node);
nosResult RegisterDelay(nosNodeFunctions* node);
nosResult RegisterArithmetic(nosNodeFunctions* node);
nosResult RegisterScalarArithmetic(nosNodeFunctions* node);
nosResult RegisterIndexOf(nosNodeFunctions* node);
nosResult RegisterIsEqual(nosNodeFunctions* node);
nosResult RegisterSetVariable(nosNodeFunctions* node);
nosResult RegisterGetVariable(nosNodeFunctions* node);

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
			GEN_CASE_NODE(MakeDynamic)
			GEN_CASE_NODE(Break)
			GEN_CASE_NODE(Indexer)
			GEN_CASE_NODE(Array)
			GEN_CASE_NODE(Delay)
			GEN_CASE_NODE(Arithmetic)
			GEN_CASE_NODE(ScalarArithmetic)
			GEN_CASE_NODE(IndexOf)
			GEN_CASE_NODE(IsEqual)
			GEN_CASE_NODE(SetVariable)
			GEN_CASE_NODE(GetVariable)
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
	outPluginFunctions->GetRenamedTypes = [](nosName* outRenamedFrom, nosName* outRenamedTo, size_t* outCount) {
		*outCount = 1;
		if (!outRenamedFrom)
			return;
		outRenamedFrom[0] = NOS_NAME("nos.fb.BinaryOperator");
		outRenamedTo[0] = NOS_NAME("nos.reflect.BinaryOperator");
	};
	outPluginFunctions->GetRenamedNodeClasses = [](nosName* outRenamedFrom, nosName* outRenamedTo, size_t* outCount) {
		constexpr auto ops = std::array{"Add", "Mul", "Sub", "Div"};
		constexpr auto types = std::array{"f32",	"f64",	 "i8",	 "i16",	  "i32",   "i64",	"u8",	"u16",
							   "u32",	"u64",	 "vec2", "vec2i", "vec2u", "vec2d", "vec3", "vec3i",
							   "vec3u", "vec3d", "vec4", "vec4i", "vec4u", "vec4d"};
		*outCount = ops.size() * types.size(); 
		if (!outRenamedFrom)
			return;
		auto idx = 0;
		for (const char* op : ops)
			for (const char* type : types)
			{
				outRenamedFrom[idx] = nos::Name("nos.math." + std::string(op) + "_" + std::string(type));
				outRenamedTo[idx] = NOS_NAME("nos.reflect.Arithmetic");
				idx++;
			}
	};
	return NOS_RESULT_SUCCESS;
}
}

}