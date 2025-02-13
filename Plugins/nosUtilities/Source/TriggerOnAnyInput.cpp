#include <Nodos/PluginHelpers.hpp>


namespace nos::utilities
{
struct TriggerOnAnyInput : NodeContext
{
	using NodeContext::NodeContext;

	nosResult Branch(nosFunctionExecuteParams* functionExecParams)
	{
		nosEngine.CallNodeFunction(NodeId, NOS_NAME("OnTrigger"));
		return NOS_RESULT_SUCCESS;
	}

	NOS_DECLARE_FUNCTIONS(
		NOS_ADD_FUNCTION(NOS_NAME("Trigger"), Branch),
	);
};


nosResult RegisterTriggerOnAnyInput(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("TriggerOnAnyInput"), TriggerOnAnyInput, out);
	return NOS_RESULT_SUCCESS;
}

}