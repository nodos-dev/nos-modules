// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include <tinyexpr.h>
#include <list>

namespace nos::math
{
struct AndNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		NodeExecuteParams pins(params);
		auto& A = *pins.GetPinData<bool>(NOS_NAME("A"));
		auto& B = *pins.GetPinData<bool>(NOS_NAME("B"));
		auto& result = *pins.GetPinData<bool>(NOS_NAME("AndResult"));
		result = A && B;
		return NOS_RESULT_SUCCESS;
	}
};

struct OrNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		NodeExecuteParams pins(params);
		auto& A = *pins.GetPinData<bool>(NOS_NAME("A"));
		auto& B = *pins.GetPinData<bool>(NOS_NAME("B"));
		auto& result = *pins.GetPinData<bool>(NOS_NAME("OrResult"));
		result = A || B;
		return NOS_RESULT_SUCCESS;
	}
};

struct NotNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		NodeExecuteParams pins(params);
		auto& X = *pins.GetPinData<bool>(NOS_NAME("X"));
		auto& notX = *pins.GetPinData<bool>(NOS_NAME("NotX"));
		auto& result = *pins.GetPinData<bool>(NOS_NAME("OrResult"));
		notX = !X;
		return NOS_RESULT_SUCCESS;
	}
};

void RegisterAnd(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.And"), AndNode, fn);
}

void RegisterOr(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Or"), OrNode, fn);
}

void RegisterNot(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Not"), NotNode, fn);
}
} // namespace nos::math

