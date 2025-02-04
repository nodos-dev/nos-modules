#include <Nodos/PluginHelpers.hpp>
#include <nosAnimationSubsystem/nosAnimationSubsystem.h>

namespace nos::animation
{
nosResult RegisterHasCustomInterpolator(nosNodeFunctions* out)
{
	out->ClassName = NOS_NAME("HasCustomInterpolator");
	out->ExecuteNode = [](void*, nosNodeExecuteParams* params) {
		NodeExecuteParams args(params);
		bool hasInterpolator;
		nosResult res = nosAnimation->HasInterpolator(args[NOS_NAME("In")].TypeName, &hasInterpolator);
		if (res != NOS_RESULT_SUCCESS)
			return res;
		nosEngine.SetPinValue(args[NOS_NAME("HasCustomInterpolator")].Id, nos::Buffer::From(hasInterpolator));
		return NOS_RESULT_SUCCESS;
	};
	return NOS_RESULT_SUCCESS;
}
} // namespace nos::animation