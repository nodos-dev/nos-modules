// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/SubsystemAPI.h>

#include <nosAnimationSubsystem/AnimEditorTypes_generated.h>
#include <PinDataAnimator.h>


NOS_INIT()
NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

namespace nos::sys::animation
{
namespace editor
{
	NOS_FBS_CREATE_FUNCTION_MAKE_FOR_UNION(nos::sys::animation::editor, FromAnimation)
}
static std::unique_ptr<struct AnimationSubsystemCtx> GAnimationSysContext;

struct AnimationSubsystemCtx
{
	AnimationSubsystemCtx() : InterpolatorManager(), Animator(InterpolatorManager), AnimationSubsystem{}
	{
		AnimationSubsystem.RegisterInterpolator = [](nosAnimationInterpolator const* interpolator) {
			nosModuleInfo callingModule{};
			auto res = nosEngine.GetCallingModule(&callingModule);
			if (res != NOS_RESULT_SUCCESS)
			{
				nosEngine.LogE("Failed to get calling module info.");
				return res;
			}
			GAnimationSysContext->InterpolatorManager.AddCustomInterpolator(
				{.name = nos::Name(callingModule.Id.Name).AsString(),
				 .version = nos::Name(callingModule.Id.Version).AsString()},
				interpolator->TypeName,
				interpolator->InterpolateCallback);
			return NOS_RESULT_SUCCESS;
		};
		AnimationSubsystem.HasInterpolator = [](nosName typeName, bool* hasHandler) {
			*hasHandler = GAnimationSysContext->InterpolatorManager.HasInterpolator(typeName);
			return NOS_RESULT_SUCCESS;
		};
		AnimationSubsystem.Interpolate =
			[](nosName typeName, const nosBuffer from, const nosBuffer to, const double t, nosBuffer* outBuf) {
				return GAnimationSysContext->InterpolatorManager.Interpolate(typeName, from, to, t, *outBuf);
			};
	}

	InterpolatorManager InterpolatorManager;
	PinDataAnimator Animator;
	nosAnimationSubsystem AnimationSubsystem;
};



nosResult OnRequest(uint32_t minorVersion, void** outSubsystemCtx)
{
	*outSubsystemCtx = &GAnimationSysContext->AnimationSubsystem;
	return NOS_RESULT_NOT_IMPLEMENTED;
}

nosResult OnPreExecuteNode(nosNodeExecuteParams* params)
{
	nosCUUID scheduledNodeId;
	if (nosEngine.GetCurrentRunnerPathInfo(&scheduledNodeId, nullptr) == NOS_RESULT_FAILED)
		return NOS_RESULT_FAILED;

	auto pathInfo = GAnimationSysContext->Animator.GetPathInfo(scheduledNodeId);

	if(!pathInfo)
		return NOS_RESULT_FAILED;

	uint64_t curFrame = pathInfo->StartFSM + pathInfo->CurFrame;

	for (size_t i = 0; i < params->PinCount; ++i)
		GAnimationSysContext->Animator.UpdatePin(params->Pins[i].Id, params->DeltaSeconds, curFrame, params->Pins[i].Data);
	return NOS_RESULT_SUCCESS;
}

void NOSAPI_CALL OnPinDeleted(nosCUUID pinId)
{
	GAnimationSysContext->Animator.OnPinDeleted(pinId);
}

nosResult ShouldExecuteNodeWithoutDirty(nosNodeExecuteParams* params)
{
	for (size_t i = 0; i < params->PinCount; ++i)
	{
		auto const& pinId = params->Pins[i].Id;
		if(params->Pins[i].ShowAs == fb::ShowAs::OUTPUT_PIN)
			continue;
		if(GAnimationSysContext->Animator.IsPinAnimating(pinId))
			return NOS_RESULT_SUCCESS;
	}
	return NOS_RESULT_NOT_FOUND;
}

void NOSAPI_CALL OnPathStart(nosCUUID scheduledPinId)
{
	nosVec2u deltaSec;
	nosEngine.GetCurrentRunnerPathInfo(nullptr, &deltaSec);
	GAnimationSysContext->Animator.CreatePathInfo(scheduledPinId, deltaSec);
}

void NOSAPI_CALL OnPathStop(nosCUUID scheduledPinId)
{
	GAnimationSysContext->Animator.DeletePathInfo(scheduledPinId);
}

void NOSAPI_CALL OnEndFrame(nosCUUID scheduledPinId, nosEndFrameCause cause)
{
	if (cause != NOS_END_FRAME_FINISHED)
		return;
	GAnimationSysContext->Animator.PathExecutionFinished(scheduledPinId);
}

void OnMessageFromEditor(uint64_t editorId, nosBuffer blob)
{
	auto msg = flatbuffers::GetRoot<editor::FromEditor>(blob.Data);
	switch (msg->event_type())
	{
	case sys::animation::editor::FromEditorUnion::AnimatePin: {
		auto animatePin = msg->event_as_AnimatePin();
		if(!animatePin || !animatePin->pin_path())
			return;
		nosCUUID pinId{};
		if (nosEngine.ItemPathToItemId(animatePin->pin_path()->c_str(), &pinId) == NOS_RESULT_SUCCESS)
		{
			nosCUUID sourceId{};
			if (nosEngine.GetSourcePinId(pinId, &sourceId) == NOS_RESULT_SUCCESS)
				GAnimationSysContext->Animator.AddAnimation(sourceId, *animatePin);
		}
		else
		{
			nosEngine.LogE("Failed to find pin %s", animatePin->pin_path()->c_str());
		}
		break;
	}
	default:
		break;
	}
}

void BroadcastAnimationTypesToEditors()
{
	auto names = GAnimationSysContext->InterpolatorManager.GetAnimatableTypes();

	editor::TAnimatableTypes types;
	for (auto& name : names)
		types.types.push_back(name.AsString());
	flatbuffers::FlatBufferBuilder fbb;
	fbb.Finish(editor::MakeFromAnimation(fbb, editor::CreateAnimatableTypes(fbb, &types)));
	nos::Buffer buf = fbb.Release();
	nosSendEditorMessageParams params{.Message = buf, .DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_BROADCAST};
	nosEngine.SendEditorMessage(&params);
}

void OnEditorConnected(uint64_t editorId)
{
	BroadcastAnimationTypesToEditors();
}

nosResult NOSAPI_CALL OnPreUnloadSubsystem()
{
	GAnimationSysContext = nullptr;
	return NOS_RESULT_SUCCESS;
}

void NOSAPI_CALL OnPostOtherModuleUnloaded(nosModuleIdentifier moduleId)
{
	bool typesChanged = GAnimationSysContext->InterpolatorManager.ModuleUnloaded(
		{ .name = nos::Name(moduleId.Name).AsString(), .version = nos::Name(moduleId.Version).AsString() });
	if (!typesChanged)
		return;
	BroadcastAnimationTypesToEditors();
}
}

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
{
	subsystemFunctions->OnRequest = nos::sys::animation::OnRequest;
	subsystemFunctions->OnPreExecuteNode = nos::sys::animation::OnPreExecuteNode;
	subsystemFunctions->ShouldExecuteNodeWithoutDirty = nos::sys::animation::ShouldExecuteNodeWithoutDirty;
	subsystemFunctions->OnPathStart = nos::sys::animation::OnPathStart;
	subsystemFunctions->OnPathStop = nos::sys::animation::OnPathStop;
	subsystemFunctions->OnEndFrame = nos::sys::animation::OnEndFrame;
	subsystemFunctions->OnPinDeleted = nos::sys::animation::OnPinDeleted;

	subsystemFunctions->OnMessageFromEditor = nos::sys::animation::OnMessageFromEditor;
	subsystemFunctions->OnEditorConnected = nos::sys::animation::OnEditorConnected;

	subsystemFunctions->OnPreUnloadSubsystem = nos::sys::animation::OnPreUnloadSubsystem;

	subsystemFunctions->OnPostOtherModuleUnloaded = nos::sys::animation::OnPostOtherModuleUnloaded;
	
	nos::sys::animation::GAnimationSysContext = std::make_unique<nos::sys::animation::AnimationSubsystemCtx>();
	return NOS_RESULT_SUCCESS;
}
}