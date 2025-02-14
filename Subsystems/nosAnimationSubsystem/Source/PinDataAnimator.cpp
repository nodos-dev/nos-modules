#include "PinDataAnimator.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/quaternion_common.hpp>


namespace nos::sys::animation
{

/// https://probablymarcus.com/blocks/2015/02/26/using-bezier-curves-as-easing-functions.html
/// Adapted from https://greweb.me/2012/02/bezier-curve-based-easing-functions-from-concept-to-implementation
/// MIT License - Copyright (c) 2012 Gaetan Renaudeau <renaudeau.gaetan@gmail.com>
struct CubicBezierEasing
{
	CubicBezierEasing(glm::vec2 p1, glm::vec2 p2) : P1(p1), P2(p2) {}

	/// x monotonically increasing from 0 to 1 with constant velocity (time)
	double Get(double x)
	{
		if (P1.x == P1.y && P2.x == P2.y)
			return x; // linear
		return Bezier(GetPercent(x), P1.y, P2.y);
	}
   
protected:
	glm::vec2 P1, P2; // Control points

	// Horner's method
	double A(double c1, double c2) { return 1.0 - 3.0 * c2 + 3.0 * c1; }
	double B(double c1, double c2) { return 3.0 * c2 - 6.0 * c1; }
	double C(double c1) { return 3.0 * c1; }

	// Return x(d) given d, x1, and x2, or y(d) given d, y1, and y2.
	// d, percentage of the distance along the curve
	double Bezier(double d, double c1, double c2) { return ((A(c1, c2) * d + B(c1, c2)) * d + C(c1)) * d; }

	double GetSlope(double d, double c1, double c2) { return 3.0 * A(c1, c2) * d * d + 2.0 * B(c1, c2) * d + C(c1); }

	double GetPercent(double x)
	{
		// Newton Raphson iteration
		// TODO: Optimize
		double guess = x;
		for (int i = 0; i < 32; ++i)
		{
			double currentSlope = GetSlope(guess, P1.x, P2.x);
			if (currentSlope == 0.0)
				return guess;
			double currentX = Bezier(guess, P1.x, P2.x) - x;
			guess -= currentX / currentSlope;
		}
		return guess;
	}
};

double EaseT(nos::fb::vec2 const& control1, nos::fb::vec2 const& control2, const double t)
{
	return CubicBezierEasing(reinterpret_cast<glm::vec2 const&>(control1), reinterpret_cast<glm::vec2 const&>(control2)).Get(t);
}

template <typename T>
requires std::is_arithmetic_v<T>
T Lerp(T start, T end, const double t)
{
	return start + (end - start) * t;
}

template <typename T, size_t Dim>
requires (Dim == 2 || Dim == 3 || Dim == 4)
T LerpVec(T start, T end, const double t)
{
	T newData{};
	newData.mutate_x(Lerp(start.x(), end.x(), t));
	newData.mutate_y(Lerp(start.y(), end.y(), t));
	if constexpr (Dim >= 3)
		newData.mutate_z(Lerp(start.z(), end.z(), t));
	if constexpr (Dim == 4)
		newData.mutate_w(Lerp(start.w(), end.w(), t));
	return newData;
}

template <typename T>
nos::Buffer ScalarInterpolator(const nosBuffer from, const nosBuffer to, const double t)
{
	T newData{};
	T start = *reinterpret_cast<T const*>(from.Data);
	T end = *reinterpret_cast<T const*>(to.Data);
	newData = Lerp(start, end, t);
	return nos::Buffer(&newData, sizeof(T));
}

template <typename T, size_t Dim>
nos::Buffer VectorInterpolator(const nosBuffer from, const nosBuffer to, const double t)
{
	T newData{};
	T start = *reinterpret_cast<T const*>(from.Data);
	T end = *reinterpret_cast<T const*>(to.Data);
	newData = LerpVec<T, Dim>(start, end, t);
	return nos::Buffer(&newData, sizeof(T));
}

template <typename T>
void AddScalarInterpolators(InterpolatorManager& interpManager, nos::Name name)
{
	interpManager.AddBuiltinInterpolator(name, ScalarInterpolator<T>);
}

template <typename T, size_t Dim>
void AddVectorInterpolators(InterpolatorManager& interpManager, nos::Name name)
{
	interpManager.AddBuiltinInterpolator(name, VectorInterpolator<T, Dim>);
}

InterpolatorManager::InterpolatorManager()
{
	AddScalarInterpolators<int>(*this, NOS_NAME("int"));
	AddScalarInterpolators<float>(*this, NOS_NAME("float"));
	AddScalarInterpolators<double>(*this, NOS_NAME("double"));
	AddScalarInterpolators<char>(*this, NOS_NAME("byte"));
	AddScalarInterpolators<short>(*this, NOS_NAME("short"));
	AddScalarInterpolators<long>(*this, NOS_NAME("long"));
	AddScalarInterpolators<long long>(*this, NOS_NAME("ulong"));
	AddScalarInterpolators<unsigned char>(*this, NOS_NAME("ubyte"));
	AddScalarInterpolators<unsigned short>(*this, NOS_NAME("ushort"));
	AddVectorInterpolators<fb::vec2, 2>(*this, NOS_NAME("nos.fb.vec2"));
	AddVectorInterpolators<fb::vec3, 3>(*this, NOS_NAME("nos.fb.vec3"));
	AddVectorInterpolators<fb::vec4, 4>(*this, NOS_NAME("nos.fb.vec4"));
	AddVectorInterpolators<fb::vec2d, 2>(*this, NOS_NAME("nos.fb.vec2d"));
	AddVectorInterpolators<fb::vec3d, 3>(*this, NOS_NAME("nos.fb.vec3d"));
	AddVectorInterpolators<fb::vec4d, 4>(*this, NOS_NAME("nos.fb.vec4d"));
	AddVectorInterpolators<fb::vec2u, 2>(*this, NOS_NAME("nos.fb.vec2u"));
	AddVectorInterpolators<fb::vec3u, 3>(*this, NOS_NAME("nos.fb.vec3u"));
	AddVectorInterpolators<fb::vec4u, 4>(*this, NOS_NAME("nos.fb.vec4u"));
	AddVectorInterpolators<fb::vec2i, 2>(*this, NOS_NAME("nos.fb.vec2i"));
	AddVectorInterpolators<fb::vec3i, 3>(*this, NOS_NAME("nos.fb.vec3i"));
	AddVectorInterpolators<fb::vec4i, 4>(*this, NOS_NAME("nos.fb.vec4i"));
	AddVectorInterpolators<fb::vec4u8, 4>(*this, NOS_NAME("nos.fb.vec4u8"));
}

void InterpolatorManager::AddBuiltinInterpolator(nos::Name name, std::function<nos::Buffer(const nosBuffer from, const nosBuffer to, const double t)> fn)
{
	std::unique_lock lock(InterpolatorsMutex);
	Interpolators[name] = [fn = std::move(fn)](const nosBuffer from, const nosBuffer to, const double t, nosBuffer* out)
		{
			*out = EngineBuffer::CopyFrom(fn(from, to, t)).Release();
			return NOS_RESULT_SUCCESS;
		};
}

void InterpolatorManager::AddCustomInterpolator(nos::fb::TModuleIdentifier moduleId, nos::Name name, InterpolatorFn fn)
{
	std::unique_lock lock(InterpolatorsMutex);
	Interpolators[name] = std::move(fn);
	ModuleToAnimators[moduleId].push_back(name);
}

bool InterpolatorManager::ModuleUnloaded(nos::fb::TModuleIdentifier moduleId)
{
	std::unique_lock lock(InterpolatorsMutex);
	auto it = ModuleToAnimators.find(moduleId);
	if (it == ModuleToAnimators.end())
		return false;
	for (auto const& name : it->second)
		Interpolators.erase(name);
	ModuleToAnimators.erase(it);
	return true;
}

nosResult InterpolatorManager::Interpolate(nos::Name typeName, const nosBuffer from, const nosBuffer to, const double t, std::optional<EngineBuffer>& outBuf)
{
	std::shared_lock lock(InterpolatorsMutex);
	auto it = Interpolators.find(typeName);
	if (it == Interpolators.end())
		return NOS_RESULT_NOT_FOUND;
	nosBuffer outNosBuf{};
	auto res = it->second(from, to, t, &outNosBuf);
	if (res == NOS_RESULT_SUCCESS)
		outBuf = EngineBuffer::FromExisting(outNosBuf);
	return res;
}

std::unordered_set<nos::Name> InterpolatorManager::GetAnimatableTypes()
{
	std::shared_lock lock(InterpolatorsMutex);
	std::unordered_set<nos::Name> types;
	for (auto const& [key, _] : Interpolators)
		types.insert(key);
	return types;
}

bool PinDataAnimator::AddAnimation(uuid const& pinId,
								   editor::AnimatePin const& buf)
{
	editor::TAnimatePin animate;
	buf.UnPackTo(&animate);
	auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
					 std::chrono::high_resolution_clock::now().time_since_epoch())
					 .count();
	AnimationData data;
	data.PinId = pinId;
	data.Duration = animate.duration;
	data.StartTime = nowMs + animate.delay;
	
	data.Interp = animate.interpolate;
	nos::TypeInfo typeInfo(pinId);
	data.TypeName = typeInfo.TypeName;

	// If constant, do not look for interpolator.
	if (data.Interp.type == editor::Interpolation::Constant)
	{
		std::unique_lock lock(AnimationsMutex);
		Animations[pinId].push(std::move(data));
		return true;
	}
	if (!InterpManager.HasInterpolator(data.TypeName))
	{
		nosEngine.LogE("No interpolator found for %s and %s.", nos::Name(typeInfo.TypeName).AsCStr(), editor::EnumNameInterpolation(data.Interp.type));
		return false;
	}
	nosEngine.LogD("Animation added for pin %s", pinId.to_string().c_str());
	std::unique_lock lock(AnimationsMutex);
	Animations[pinId].push(std::move(data));
	return true;
}

void PinDataAnimator::UpdatePin(uuid const& pinId, 
								nosVec2u const& deltaSeconds,
								uint64_t curFSM,
								const nosBuffer* currentData)
{	
	std::shared_lock lock(AnimationsMutex);
	auto it = Animations.find(pinId);
	if (it == Animations.end())
		return;
	auto& animQueue = it->second;
	if (animQueue.empty())
	{
		lock.unlock();
		std::unique_lock ulock(AnimationsMutex);
		Animations.erase(pinId);
		return;
	}
	const AnimationData& animData = animQueue.top();
	int64_t diff = curFSM - MillisecondsToFrameNumber(animData.StartTime, deltaSeconds);
	if (diff < 0)
		return;
	if (animData.Started == false)
	{

		switch (animData.Interp.type)
		{
		case editor::Interpolation::Lerp: {
			auto* lerp = const_cast<editor::TLerp*>(animData.Interp.AsLerp());
			if (lerp->start.empty())
				lerp->start = std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(currentData->Data), reinterpret_cast<uint8_t const*>(currentData->Data) + currentData->Size);
			break;
		}
		case editor::Interpolation::CubicBezier: {
			auto* cubic = const_cast<editor::TCubicBezier*>(animData.Interp.AsCubicBezier());
			if (cubic->start.empty())
				cubic->start = std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(currentData->Data), reinterpret_cast<uint8_t const*>(currentData->Data) + currentData->Size);
			break;
		}
		default: break;
		}
		const_cast<AnimationData&>(animData).Started = true;
	}
	const double t = glm::clamp(static_cast<double>(diff) / MillisecondsToFrameNumber(animData.Duration, deltaSeconds), 0.0, 1.0);
	nosResult result = NOS_RESULT_FAILED;
	if (t >= 0.0)
	{
		if (animData.Interp.type == editor::Interpolation::Constant)
		{
			nosEngine.SetPinValue(pinId,
								  nosBuffer{.Data = (void*)animData.Interp.AsConstant()->value.data(),
											.Size = animData.Interp.AsConstant()->value.size()});
			result = NOS_RESULT_SUCCESS;
		}
		else
		{
			double interpolationT = t;
			if (animData.Interp.type == editor::Interpolation::CubicBezier)
				interpolationT = EaseT(animData.Interp.AsCubicBezier()->control1, animData.Interp.AsCubicBezier()->control2, t);
			nosBuffer start{};
			nosBuffer end{};
			switch (animData.Interp.type) 
			{
			case editor::Interpolation::CubicBezier:
			{
				auto* cubic = animData.Interp.AsCubicBezier();
				start = nosBuffer((void*)cubic->start.data(), cubic->start.size());
				end = nosBuffer((void*)cubic->end.data(), cubic->end.size());
				break;
			}
			case editor::Interpolation::Lerp:
			{
				auto* lerp = animData.Interp.AsLerp();
				start = nosBuffer((void*)lerp->start.data(), lerp->start.size());
				end = nosBuffer((void*)lerp->end.data(), lerp->end.size());
				break;
			}
			}
			std::optional<EngineBuffer> buf;
			result = InterpManager.Interpolate(animData.TypeName, start, end, interpolationT, buf);
			if (result == NOS_RESULT_SUCCESS && buf)
				nosEngine.SetPinValue(pinId, *buf); 
		}
	}
	if (result != NOS_RESULT_SUCCESS)
		nosEngine.LogE("Failed to animate pin %s", pinId.to_string().c_str());
	if (t >= 1.0 || result != NOS_RESULT_SUCCESS)
	{
		lock.unlock();
		std::unique_lock ulock(AnimationsMutex);
		it = Animations.find(pinId);
		if (it == Animations.end())
			return;
		auto& animQueue = it->second;
		animQueue.pop();
		if (animQueue.empty())
			Animations.erase(it);
	}
}

bool PinDataAnimator::IsPinAnimating(uuid const& pinId)
{
	std::shared_lock lock(AnimationsMutex);
	return Animations.contains(pinId);
}

void PinDataAnimator::OnPinDeleted(uuid const& pinId)
{
	std::unique_lock lock(AnimationsMutex);
	Animations.erase(pinId);
}

std::optional<PathInfo> PinDataAnimator::GetPathInfo(uuid const& scheduledNodeId) 
{
	std::shared_lock lock(AnimationsMutex);
	auto it = PathInfos.find(scheduledNodeId);
	if (it == PathInfos.end())
		return std::nullopt;
	return it->second;
}

void PinDataAnimator::CreatePathInfo(uuid const& scheduledNodeId, nosVec2u const& deltaSec)
{
	std::unique_lock lock(PathInfosMutex);
	uint64_t startFSM = MillisecondsToFrameNumber(std::chrono::duration_cast<std::chrono::milliseconds>(
													  std::chrono::high_resolution_clock::now().time_since_epoch())
														  .count(),
														  deltaSec);
	PathInfos[scheduledNodeId] = {.StartFSM = startFSM};
}

void PinDataAnimator::DeletePathInfo(uuid const& scheduledNodeId)
{
	std::unique_lock lock(PathInfosMutex);
	PathInfos.erase(scheduledNodeId);
}

void PinDataAnimator::PathExecutionFinished(uuid const& scheduledNodeId)
{
	std::unique_lock lock(PathInfosMutex);
	PathInfos[scheduledNodeId].CurFrame++;
}

} // namespace nos::sys::anim
