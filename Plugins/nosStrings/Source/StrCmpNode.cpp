// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

namespace nos::strings
{
struct IsSameStringNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		auto pin = GetPinValues(params);
		auto firstStr = GetPinValue<const char>(pin, NOS_NAME("First"));
		auto secondStr = GetPinValue<const char>(pin, NOS_NAME("Second"));
		SetPinValue(NOS_NAME("IsSame"), nos::Buffer::From(strcmp(firstStr, secondStr) == 0));
		return NOS_RESULT_SUCCESS;
	}
};


nosResult RegisterIsSameString(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("IsSameString"), IsSameStringNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos