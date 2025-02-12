#include <Nodos/PluginHelpers.hpp>


namespace nos::utilities
{
struct ConditionalTrigger : NodeContext
{
	using NodeContext::NodeContext;

	nosResult Branch(nosFunctionExecuteParams* functionExecParams)
	{
		NodeExecuteParams params(functionExecParams->FunctionNodeExecuteParams);
		bool condition = *params.GetPinData<bool>(NOS_NAME("Condition"));
		if (condition)
			nosEngine.CallNodeFunction(NodeId, NOS_NAME("True"));
		else
			nosEngine.CallNodeFunction(NodeId, NOS_NAME("False"));
		return NOS_RESULT_SUCCESS;
	}

	NOS_DECLARE_FUNCTIONS(
		NOS_ADD_FUNCTION(NOS_NAME("Branch"), Branch),
	);
};


nosResult RegisterConditionalTrigger(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("ConditionalTrigger"), ConditionalTrigger, out);
	return NOS_RESULT_SUCCESS;
}

}