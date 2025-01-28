#include <Nodos/PluginHelpers.hpp>


namespace nos::animation
{

struct AnimateNode : NodeContext
{
	using NodeContext::NodeContext;
	bool Running = false;
	uint64_t AnimationFrameCount = 0;

	nosResult ExecuteNode(nosNodeExecuteParams* params)
	{
		NodeExecuteParams args(params);
		if (Running)
		{
			float animationDuration = *args.GetPinData<float>(NOS_NAME("Duration"));
			float out = (params->DeltaSeconds.x * AnimationFrameCount) / (params->DeltaSeconds.y * animationDuration);
			if (out <= 1.0f)
			{
				SetPinValue(NOS_NAME("t"), nos::Buffer::From(out));
				AnimationFrameCount++;
				return NOS_RESULT_SUCCESS;
			}
			else
				Running = false;
		}
		params->MarkAllOutsDirty = false;
		return NOS_RESULT_SUCCESS;
	}

	nosResult Start(nosFunctionExecuteParams* functionExecParams)
	{
		AnimationFrameCount = 0;
		Running = true;
		return NOS_RESULT_SUCCESS;
	}

	static nosResult GetFunctions(size_t* outCount, nosName* outFunctionNames, nosPfnNodeFunctionExecute* outFunctions) 
	{
		if (!outFunctions)
		{
			*outCount = 1;
			return NOS_RESULT_SUCCESS;
		}
		
		outFunctionNames[0] = NOS_NAME("Start");
		outFunctions[0] = [](void* ctx, nosFunctionExecuteParams* functionExecParams) -> nosResult
			{
				return static_cast<AnimateNode*>(ctx)->Start(functionExecParams);
			};

		return NOS_RESULT_SUCCESS; 
	}
};


nosResult RegisterAnimate(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Animate"), AnimateNode, out);
	return NOS_RESULT_SUCCESS;
}

}