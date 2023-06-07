#include "Track.h"

namespace mz
{

MZ_REGISTER_NAME(Camera_Position_Name, "Camera Position");
MZ_REGISTER_NAME(Camera_Orientation_Name, "Camera Orientation");
MZ_REGISTER_NAME(Nodal_Offset_Name, "Nodal Offset");
MZ_REGISTER_NAME(Nodal_Point_Name, "Nodal Point");

void RegisterCalculateNodalPoint(mzNodeFunctions& out)
{
	out.TypeName = MZ_NAME_STATIC("mz.track.CalculateNodalPoint");
	out.ExecuteNode = [](auto ctx, auto args) {
		auto pins = GetPinValues(args);
		auto pos = (glm::dvec3*)pins[Camera_Position_Name];
		auto rot = (glm::dvec3*)pins[Camera_Orientation_Name];
		auto sca = (f64*)pins[Nodal_Offset_Name];
		auto out = (glm::dvec3*)pins[Nodal_Point_Name];
		glm::dvec2 ANG = glm::radians(glm::dvec2(rot->z, rot->y));
		glm::dvec2 COS = cos(ANG);
		glm::dvec2 SIN = sin(ANG);
		glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
		*out = *pos + f **sca;
		return MZ_RESULT_SUCCESS;
	};
}
}
