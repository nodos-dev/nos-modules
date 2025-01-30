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
			if(AnimationFrameCount == 0)
				nosEngine.CallNodeFunction(NodeId, NOS_NAME("Started_Internal"));
			float animationDuration = *args.GetPinData<float>(NOS_NAME("Duration"));
			float out = (params->DeltaSeconds.x * AnimationFrameCount) / (params->DeltaSeconds.y * animationDuration);
			AnimationFrameCount++;
			bool loop = *args.GetPinData<bool>(NOS_NAME("Loop"));
			bool finished = false;
			if (loop)
			{
				bool swing = *args.GetPinData<bool>(NOS_NAME("Swing"));
				int64_t loops = static_cast<int64_t>(out);
				bool isEven = loops % 2 == 0;
				if (swing && !isEven)
					out = loops + 1 - out;
				else
					out = out - loops;
			}
			else if (out >= 1.0f)
			{
				out = 1.0f;
				finished = true;
			}
			SetPinValue(NOS_NAME("t"), nos::Buffer::From(out));
			if (finished)
			{
				nosEngine.CallNodeFunction(NodeId, NOS_NAME("Finished_Internal"));
				Running = false;
			}
			return NOS_RESULT_SUCCESS;
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

	nosResult Pause(nosFunctionExecuteParams* functionExecParams)
	{
		Running = false;
		return NOS_RESULT_SUCCESS;
	}

	nosResult Continue(nosFunctionExecuteParams* functionExecParams)
	{
		Running = true;
		return NOS_RESULT_SUCCESS;
	}

	NOS_DECLARE_FUNCTIONS(
		NOS_ADD_FUNCTION(NOS_NAME("Start"), Start),
		NOS_ADD_FUNCTION(NOS_NAME("Pause"), Pause),
		NOS_ADD_FUNCTION(NOS_NAME("Continue"), Continue)
	);
};


nosResult RegisterAnimate(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Animate"), AnimateNode, out);
	return NOS_RESULT_SUCCESS;
}

}