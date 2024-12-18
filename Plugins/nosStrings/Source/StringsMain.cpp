// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

// Includes
#include <Nodos/PluginHelpers.hpp>

NOS_INIT_WITH_MIN_REQUIRED_MINOR(9)

NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

namespace nos::strings
{

enum Nodes : int
{	// CPU nodes
	IsSameString = 0,
	Regex,
	Pin2Json,
	Json2Pin,
	Tokenizer,
	Count
};

nosResult RegisterIsSameString(nosNodeFunctions*);
nosResult RegisterRegex(nosNodeFunctions*);
nosResult RegisterPin2Json(nosNodeFunctions*);
nosResult RegisterJson2Pin(nosNodeFunctions*);
nosResult RegisterTokenizer(nosNodeFunctions*);

nosResult NOSAPI_CALL ExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	*outSize = Nodes::Count;
	if (!outList)
		return NOS_RESULT_SUCCESS;

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
			GEN_CASE_NODE(IsSameString)
			GEN_CASE_NODE(Regex)
			GEN_CASE_NODE(Pin2Json)
			GEN_CASE_NODE(Json2Pin)
			GEN_CASE_NODE(Tokenizer)
		}
	}
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportPlugin(nosPluginFunctions* out)
{
	out->ExportNodeFunctions = ExportNodeFunctions;
	out->GetRenamedTypes = [](nosName* outRenamedFrom, nosName* outRenamedTo, size_t* outSize) {
		if (!outRenamedFrom)
		{
			*outSize = 4;
			return;
		}
		// clang-format off
		outRenamedFrom[0] = NOS_NAME("nos.utilities.Regex"); outRenamedTo[0] = NOS_NAME("nos.str.Regex");
		outRenamedFrom[1] = NOS_NAME("nos.utilities.Pin2Json"); outRenamedTo[1] = NOS_NAME("nos.str.Pin2Json");
		outRenamedFrom[2] = NOS_NAME("nos.utilities.Json2Pin"); outRenamedTo[2] = NOS_NAME("nos.str.Json2Pin");
		outRenamedFrom[3] = NOS_NAME("nos.utilities.IsSameString"); outRenamedTo[3] = NOS_NAME("nos.str.IsSameString");
		// clang-format on
	};
	return NOS_RESULT_SUCCESS;
}
}
}	
