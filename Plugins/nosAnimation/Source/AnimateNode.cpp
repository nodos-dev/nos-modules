#include <Nodos/PluginHelpers.hpp>


namespace nos::animation
{
NOS_REGISTER_NAME(t)
struct AnimateNode : NodeContext
{
	using NodeContext::NodeContext;
	bool StartNextFrame = false;
	bool Running = false;
	int64_t AnimationFrameCount = 0;

	nosResult ExecuteNode(nosNodeExecuteParams* params)
	{
		NodeExecuteParams args(params);
		if (StartNextFrame)
		{
			StartNextFrame = false;
			AnimationFrameCount = 0;
			Running = true;
			nosEngine.CallNodeFunction(NodeId, NOS_NAME("Started_Internal"));
		}
		if (Running)
		{
			bool reverse = *args.GetPinData<bool>(NOS_NAME("Reverse"));
			double animationDuration = *args.GetPinData<double>(NOS_NAME("Duration"));
			double out = double(params->DeltaSeconds.x * AnimationFrameCount) / double(params->DeltaSeconds.y * animationDuration);
			bool loop = *args.GetPinData<bool>(NOS_NAME("Loop"));
			bool finished = false;
			if (loop)
			{
				bool swing = *args.GetPinData<bool>(NOS_NAME("Swing"));
				int64_t loops = std::floor(out);
				bool isEven = loops % 2 == 0;
				if (swing && !isEven)
					out = loops + 1 - out;
				else
					out = out - loops;
			}
			else
			{
				if (!reverse && out >= 1.0f)
				{
					out = 1.0f;
					finished = true;
				}
				if (reverse && out <= 0.0f)
				{
					AnimationFrameCount = 0;
					out = 0.0f;
					finished = true;
				}
			}
			SetPinValue(NSN_t, nos::Buffer::From(out));
			if (finished)
			{
				nosEngine.CallNodeFunction(NodeId, NOS_NAME("Finished_Internal"));
				Running = false;
			}
			else
				AnimationFrameCount = AnimationFrameCount + (reverse ? -1 : 1);
			return NOS_RESULT_SUCCESS;
		}
		params->MarkAllOutsDirty = false;
		return NOS_RESULT_SUCCESS;
	}

	nosResult Start(nosFunctionExecuteParams* functionExecParams)
	{
		StartNextFrame = true;
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

	nosResult Reset(nosFunctionExecuteParams* functionExecParams)
	{
		AnimationFrameCount = 0;
		Running = false;
		double out = 0.0f;
		SetPinValue(NSN_t, nos::Buffer::From(out));
		return NOS_RESULT_SUCCESS;
	}

	NOS_DECLARE_FUNCTIONS(
		NOS_ADD_FUNCTION(NOS_NAME("Start"), Start),
		NOS_ADD_FUNCTION(NOS_NAME("Pause"), Pause),
		NOS_ADD_FUNCTION(NOS_NAME("Continue"), Continue),
		NOS_ADD_FUNCTION(NOS_NAME("Reset"), Reset)
	);
};


nosResult RegisterAnimate(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Animate"), AnimateNode, out);
	return NOS_RESULT_SUCCESS;
}

}