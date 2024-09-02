// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

namespace nos::utilities
{
NOS_REGISTER_NAME_SPACED(Nos_Utilities_PropagateExecution, "nos.utilities.PropagateExecution")
struct PropagateExecutionNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nosEngine.CallNodeFunction(NodeId, NOS_NAME("Propagate"));
		nosScheduleNodeParams schedule{
			.NodeId = NodeId,
			.AddScheduleCount = 1
		};
		nosEngine.ScheduleNode(&schedule);
		return NOS_RESULT_SUCCESS;
	}

	void OnPathStart() override
	{
		nosScheduleNodeParams schedule{
			.NodeId = NodeId,
			.AddScheduleCount = 1
		};
		nosEngine.ScheduleNode(&schedule);
	}
};

nosResult RegisterPropagateExecution(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NSN_Nos_Utilities_PropagateExecution, PropagateExecutionNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities