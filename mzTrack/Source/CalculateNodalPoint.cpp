#include "Track.h"

namespace mz
{

MZ_REGISTER_NAME_SPACED(Camera_Position, "Camera Position");
MZ_REGISTER_NAME_SPACED(Camera_Orientation, "Camera Orientation");
MZ_REGISTER_NAME_SPACED(Nodal_Offset, "Nodal Offset");
MZ_REGISTER_NAME_SPACED(Nodal_Point, "Nodal Point");

void RegisterCalculateNodalPoint(mzNodeFunctions& out)
{
	out.TypeName = MZ_NAME_STATIC("mz.track.CalculateNodalPoint");
	out.ExecuteNode = [](auto ctx, auto args) {
		auto pins = GetPinValues(args);
		auto pos = (glm::dvec3*)pins[MZN_Camera_Position];
		auto rot = (glm::dvec3*)pins[MZN_Camera_Orientation];
		auto sca = (f64*)pins[MZN_Nodal_Offset];
		auto out = (glm::dvec3*)pins[MZN_Nodal_Point];
		glm::dvec2 ANG = glm::radians(glm::dvec2(rot->z, rot->y));
		glm::dvec2 COS = cos(ANG);
		glm::dvec2 SIN = sin(ANG);
		glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
		*out = *pos + f **sca;
		return MZ_RESULT_SUCCESS;
	};
}
}
