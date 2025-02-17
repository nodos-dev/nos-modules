// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/Helpers.hpp>
#include "Track_generated.h"
#include <glm/glm.hpp>

namespace nos::track
{
void RegisterAddTrack(nosNodeFunctions* funcs) { 
	funcs->ClassName = NOS_NAME("AddTrack");
	funcs->ExecuteNode = [](void*, nosNodeExecuteParams* params) {
		auto pins = GetPinValues(params);
		auto ids = GetPinIds(params);
		// TODO: Remove these once generic table aritmetic ops are supported
		auto* xTrack = flatbuffers::GetMutableRoot<track::Track>(pins[NOS_NAME("X")]);
		auto* yTrack = flatbuffers::GetMutableRoot<track::Track>(pins[NOS_NAME("Y")]);
		track::TTrack sumTrack;
		xTrack->UnPackTo(&sumTrack);
		reinterpret_cast<glm::vec3&>(sumTrack.location) += reinterpret_cast<const glm::vec3&>(*yTrack->location());
		reinterpret_cast<glm::vec3&>(sumTrack.rotation) += reinterpret_cast<const glm::vec3&>(*yTrack->rotation());
		sumTrack.fov += yTrack->fov();
		sumTrack.focus += yTrack->focus();
		sumTrack.zoom += yTrack->zoom();
		sumTrack.render_ratio += yTrack->render_ratio();
		reinterpret_cast<glm::vec2&>(sumTrack.sensor_size) +=
			reinterpret_cast<const glm::vec2&>(*yTrack->sensor_size());
		sumTrack.pixel_aspect_ratio += yTrack->pixel_aspect_ratio();
		sumTrack.nodal_offset += yTrack->nodal_offset();
		sumTrack.focus_distance += yTrack->focus_distance();
		auto& sumDistortion = sumTrack.lens_distortion;
		auto& yDistortion = *yTrack->lens_distortion();
		reinterpret_cast<glm::vec2&>(sumDistortion.mutable_center_shift()) +=
			reinterpret_cast<const glm::vec2&>(yDistortion.center_shift());
		reinterpret_cast<glm::vec2&>(sumDistortion.mutable_k1k2()) +=
			reinterpret_cast<const glm::vec2&>(yDistortion.k1k2());
		sumDistortion.mutate_distortion_scale(sumDistortion.distortion_scale() + yDistortion.distortion_scale());
		return nosEngine.SetPinValue(ids[NOS_NAME("Z")], nos::Buffer::From(sumTrack));
	};
}
}