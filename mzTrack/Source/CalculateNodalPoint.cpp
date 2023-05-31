#include "Track.h"

namespace mz
{
void RegisterCalculateNodalPoint(mzNodeFunctions& out)
{
	out.TypeName = "mz.track.CalculateNodalPoint";
	out.ExecuteNode = [](auto ctx, auto args) {
		auto pins = GetPinValues(args);
		auto pos = (glm::dvec3*)pins["Camera Position"];
		auto rot = (glm::dvec3*)pins["Camera Orientation"];
		auto sca = (f64*)pins["Nodal Offset"];
		auto out = (glm::dvec3*)pins["Nodal Point"];
		glm::dvec2 ANG = glm::radians(glm::dvec2(rot->z, rot->y));
		glm::dvec2 COS = cos(ANG);
		glm::dvec2 SIN = sin(ANG);
		glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
		*out = *pos + f **sca;
		return MZ_RESULT_SUCCESS;
	};
}
}
