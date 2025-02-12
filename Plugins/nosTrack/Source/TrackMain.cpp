// Copyright MediaZ Teknoloji A.S. All Rights Reserved.
#include "Track.h"
#include "nosAnimationSubsystem/nosAnimationSubsystem.h"
#include "Track_generated.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/ext/quaternion_common.hpp>

NOS_INIT()
NOS_ANIMATION_INIT()

NOS_BEGIN_IMPORT_DEPS()
	NOS_ANIMATION_IMPORT()
NOS_END_IMPORT_DEPS()

namespace nos::track
{

enum TrackNode : int
{
	FreeD,
	UserTrack,
	AddTrack,
	Count
};

void RegisterFreeDNode(nosNodeFunctions* functions);
void RegisterController(nosNodeFunctions* functions);
void RegisterAddTrack(nosNodeFunctions*);

nosResult NOSAPI_CALL ExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	*outSize = (size_t)TrackNode::Count;
	if (!outList)
		return NOS_RESULT_SUCCESS;

	for (int i = 0; i < TrackNode::Count; ++i)
	{
		auto node = outList[i];
		switch ((TrackNode)i)
		{
		case TrackNode::FreeD:
			RegisterFreeDNode(node);
			break;
		case TrackNode::UserTrack:
			RegisterController(node);
			break;
		case TrackNode::AddTrack:
			RegisterAddTrack(node); 
			break;
		}
	}
	return NOS_RESULT_SUCCESS;
}

template <typename T, size_t N>
T LerpVec(const T& from, const T& to, double t)
{ 
	T result{};
	result.mutate_x(glm::mix(from.x(), to.x(), t));
	result.mutate_y(glm::mix(from.y(), to.y(), t));
	if constexpr (N > 2)
		result.mutate_z(glm::mix(from.z(), to.z(), t));
	return result;
}

template <typename Vec3Type>
Vec3Type InterpolateRotation(const Vec3Type& start, const Vec3Type& end, const double t)
{
	using ScalarType = decltype(start.x());
	glm::vec<3, ScalarType> glmStart(start.x(), start.y(), start.z());
	glm::vec<3, ScalarType> glmEnd(end.x(), end.y(), end.z());
	glm::qua<ScalarType> startQuat = glm::quat(glm::radians(glmStart));
	glm::qua<ScalarType> endQuat = glm::quat(glm::radians(glmEnd));
	glm::qua<ScalarType> newQuat = glm::slerp(startQuat, endQuat, static_cast<ScalarType>(t));
	glm::vec<3, ScalarType> newEuler = glm::degrees(glm::eulerAngles(newQuat));
	return Vec3Type(newEuler.x, newEuler.y, newEuler.z);
}

nosResult NOSAPI_CALL InterpolateTrack(const nosBuffer from, const nosBuffer to, const double t, nosBuffer* out)
{
	auto start = flatbuffers::GetRoot<track::Track>(from.Data);
	auto end = flatbuffers::GetRoot<track::Track>(to.Data);
	nos::track::TTrack interm;
	interm.location = LerpVec<fb::vec3, 3>(*start->location(), *end->location(), t);
	interm.rotation = InterpolateRotation(*start->rotation(), *end->rotation(), t); // quat
	interm.fov = glm::mix(start->fov(), end->fov(), t);
	interm.focus = glm::mix(start->focus(), end->focus(), t);
	interm.zoom = glm::mix(start->zoom(), end->zoom(), t);
	interm.render_ratio = glm::mix(start->render_ratio(), end->render_ratio(), t);
	interm.sensor_size = LerpVec<fb::vec2, 2>(*start->sensor_size(), *end->sensor_size(), t);
	interm.pixel_aspect_ratio = glm::mix(start->pixel_aspect_ratio(), end->pixel_aspect_ratio(), t);
	interm.nodal_offset = glm::mix(start->nodal_offset(), end->nodal_offset(), t);
	interm.focus_distance = glm::mix(start->focus_distance(), end->focus_distance(), t);
	auto distortionStart = *start->lens_distortion();
	auto distortionEnd = *end->lens_distortion();
	auto& distortion = interm.lens_distortion;
	distortion.mutable_center_shift() =
		LerpVec<fb::vec2, 2>(distortionStart.center_shift(), distortionEnd.center_shift(), t);
	distortion.mutable_k1k2() = LerpVec<fb::vec2, 2>(distortionStart.k1k2(), distortionEnd.k1k2(), t);
	distortion.mutate_distortion_scale(glm::mix(distortionStart.distortion_scale(), distortionEnd.distortion_scale(), t));

	nos::Table<track::Track> result = nos::Buffer::From(interm);
	nosEngine.AllocateBuffer(result.Size(), out);
	memcpy(out->Data, result.Data(), result.Size());
	return NOS_RESULT_SUCCESS;
}

nosResult NOSAPI_CALL InterpolateTransform(const nosBuffer from, const nosBuffer to, const double t, nosBuffer* out)
{
	auto start = flatbuffers::GetRoot<fb::Transform>(from.Data);
	auto end = flatbuffers::GetRoot<fb::Transform>(to.Data);
	nos::fb::Transform interm;
	interm.mutable_position() = LerpVec<fb::vec3d, 3>(start->position(), end->position(), t);
	interm.mutable_rotation() = InterpolateRotation(start->rotation(), end->rotation(), t); // quat
	interm.mutable_scale() = LerpVec<fb::vec3d, 3>(start->scale(), end->scale(), t);

	nosEngine.AllocateBuffer(sizeof(interm), out);
	memcpy(out->Data, &interm, sizeof(interm));
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportPlugin(nosPluginFunctions* outFunctions)
{
	outFunctions->ExportNodeFunctions = ExportNodeFunctions;
	outFunctions->GetRenamedTypes = [](nosName* outRenamedFrom, nosName* outRenamedTo, size_t* outCount) {
		*outCount = 4;
		if (!outRenamedFrom || !outRenamedTo)
			return;
		outRenamedFrom[0] = NOS_NAME_STATIC("nos.fb.Track");
		outRenamedTo[0] = NOS_NAME_STATIC(track::Track::GetFullyQualifiedName());
		outRenamedFrom[1] = NOS_NAME_STATIC("nos.fb.LensDistortion");
		outRenamedTo[1] = NOS_NAME_STATIC(track::LensDistortion::GetFullyQualifiedName());
		outRenamedFrom[2] = NOS_NAME_STATIC("nos.fb.CoordinateSystem");
		outRenamedTo[2] = NOS_NAME_STATIC("nos.track.CoordinateSystem");
		outRenamedFrom[3] = NOS_NAME_STATIC("nos.fb.RotationSystem");
		outRenamedTo[3] = NOS_NAME_STATIC("nos.track.RotationSystem");
	};


	nosAnimationInterpolator trackInterpolator = {
		.TypeName = NOS_NAME(track::Track::GetFullyQualifiedName()), .InterpolateCallback = InterpolateTrack};
	nosAnimation->RegisterInterpolator(&trackInterpolator);

	nosAnimationInterpolator transformInterpolator = {.TypeName = NOS_NAME(fb::Transform::GetFullyQualifiedName()),
											 .InterpolateCallback = InterpolateTransform};
	nosAnimation->RegisterInterpolator(&trackInterpolator);


	return NOS_RESULT_SUCCESS;
}
}

} // namespace nos::track