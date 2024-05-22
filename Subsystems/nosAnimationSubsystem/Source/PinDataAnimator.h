/*
 * Copyright MediaZ AS. All Rights Reserved.
 */

#pragma once

#include <Nodos/SubsystemAPI.h>
#include <nosAnimationSubsystem/AnimEditorTypes_generated.h>
#include <Nodos/Helpers.hpp>

namespace nos::sys::anim
{

inline uint64_t MillisecondsToFrameNumber(uint64_t ms, nosVec2u deltaSeconds)
{
	return (ms * (uint64_t)deltaSeconds.y) / (uint64_t)deltaSeconds.x / 1000ull;
}

struct AnimationData
{
	nosUUID PinId;
	nos::Name TypeName;
	editor::InterpolationUnion Interp;
	uint64_t StartTime; 
	uint64_t Duration;
	bool Started = false;
};

struct PathInfo
{
	uint64_t StartFSM;
	uint64_t CurFrame;
};

struct PinDataAnimator
{
	struct InterpolatorKey
	{
		editor::Interpolation Mode;
		nos::Name TypeName;
		InterpolatorKey(editor::Interpolation mode, nos::Name typeName) : Mode(mode), TypeName(typeName) {}
		struct Hash
		{
			size_t operator()(InterpolatorKey const& t) const
			{
				size_t re = 0;
				nos::hash_combine(re, t.Mode, t.TypeName.ID);
				return re;
			}
		};

		bool operator==(InterpolatorKey const& rhs) const = default;
	};
	using InterpolatorFn = std::function<nos::Buffer(editor::InterpolationUnion const&, const double t)>;
	using InterpolatorMap = std::unordered_map<InterpolatorKey, InterpolatorFn, InterpolatorKey::Hash>;

	PinDataAnimator();
	bool AddAnimation(nosUUID const& pinId,
					  editor::AnimatePin const& animate);
	void UpdatePin(nosUUID const& pinId, nosVec2u const& deltaSeconds, uint64_t curFSM, const nosBuffer* currentData);
	bool IsPinAnimating(nosUUID const& pinId);
	void OnPinDeleted(nosUUID const& pinId);
	std::unordered_set<nos::Name> GetAnimatableTypes();
	std::optional<PathInfo> GetPathInfo(nosUUID const& nodeId);

	void CreatePathInfo(nosUUID const& scheduledNodeId, nosVec2u const& deltaSec);
	void DeletePathInfo(nosUUID const& scheduledNodeId);
	void PathExecutionFinished(nosUUID const& scheduledNodeId);

	template <editor::Interpolation Mode, tmp::StrLiteral TypeName>
	void AddInterpolator(InterpolatorFn fn)
	{
		Interpolators[InterpolatorKey(Mode, nos::Name::GetName<TypeName>())] = std::move(fn);
	}

	
	struct TimeAscending
	{
		bool operator()(AnimationData const& lhs, AnimationData const& rhs) const
		{
			return lhs.StartTime > rhs.StartTime;
		}
	};

	std::shared_mutex PathInfosMutex;
	std::unordered_map<nosUUID, PathInfo> PathInfos;

	std::shared_mutex AnimationsMutex;
	std::unordered_map<nosUUID, std::priority_queue<AnimationData, std::vector<AnimationData>, TimeAscending>> Animations;
	InterpolatorMap Interpolators;
};

} // namespace nos::engine
