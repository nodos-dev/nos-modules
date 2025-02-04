/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once

#include <Nodos/SubsystemAPI.h>
#include <nosAnimationSubsystem/AnimEditorTypes_generated.h>
#include <Nodos/Helpers.hpp>
#include "nosAnimationSubsystem/nosAnimationSubsystem.h"

namespace nos::sys::animation
{

inline uint64_t MillisecondsToFrameNumber(uint64_t ms, nosVec2u deltaSeconds)
{
	return (ms * (uint64_t)deltaSeconds.y) / (uint64_t)deltaSeconds.x / 1000ull;
}

struct InterpolatorManager
{
	InterpolatorManager();
	using InterpolatorFn = std::function<nosResult(const nosBuffer from, const nosBuffer to, const double t, nosBuffer* outBuf)>;
	void AddBuiltinInterpolator(nos::Name name, std::function<nos::Buffer(const nosBuffer from, const nosBuffer to, const double t)> fn);

	void AddCustomInterpolator(nos::fb::TModuleIdentifier moduleId, nos::Name name, InterpolatorFn fn);

	// Returns true if any interpolator was removed
	bool ModuleUnloaded(nos::fb::TModuleIdentifier moduleId);

	bool HasInterpolator(nos::Name name)
	{
		std::shared_lock lock(InterpolatorsMutex);
		return Interpolators.contains(name);
	}

	nosResult Interpolate(nos::Name typeName, const nosBuffer from, const nosBuffer to, const double t, nosBuffer& outBuf);

	std::unordered_set<nos::Name> GetAnimatableTypes();

	std::shared_mutex InterpolatorsMutex;
	std::unordered_map<nos::Name, InterpolatorFn> Interpolators;
	std::unordered_map<nos::fb::TModuleIdentifier, std::vector<nos::Name>> ModuleToAnimators;
};

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
	PinDataAnimator(InterpolatorManager& interpManager) : InterpManager(interpManager) {}

	bool AddAnimation(nosUUID const& pinId,
					  editor::AnimatePin const& animate);
	void UpdatePin(nosUUID const& pinId, nosVec2u const& deltaSeconds, uint64_t curFSM, const nosBuffer* currentData);
	bool IsPinAnimating(nosUUID const& pinId);
	void OnPinDeleted(nosUUID const& pinId);
	std::optional<PathInfo> GetPathInfo(nosUUID const& nodeId);

	void CreatePathInfo(nosUUID const& scheduledNodeId, nosVec2u const& deltaSec);
	void DeletePathInfo(nosUUID const& scheduledNodeId);
	void PathExecutionFinished(nosUUID const& scheduledNodeId);

	struct TimeAscending
	{
		bool operator()(AnimationData const& lhs, AnimationData const& rhs) const
		{
			return lhs.StartTime > rhs.StartTime;
		}
	};

	InterpolatorManager& InterpManager;

	std::shared_mutex PathInfosMutex;
	std::unordered_map<nosUUID, PathInfo> PathInfos;

	std::shared_mutex AnimationsMutex;
	std::unordered_map<nosUUID, std::priority_queue<AnimationData, std::vector<AnimationData>, TimeAscending>> Animations;

};

} // namespace nos::engine
