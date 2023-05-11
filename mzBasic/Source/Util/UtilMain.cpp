#include <MediaZ/Plugin.h>
#include "Args.h"
#include "Builtins_generated.h"

#include <chrono>
#include <glm/glm.hpp>

#include "ImageIO/ImageIO.inl"
#include "ChannelViewer.inl"
#include "Switcher/Switcher.inl"

namespace mz
{

class TimeNodeContext
{
public:
	TimeNodeContext() : TimeStart(std::chrono::high_resolution_clock::now()) {}

	void Update() {}

	float GetDeltaTime()
	{
		auto now = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now - TimeStart).count() / 1000.f;
	}

private:
	std::chrono::high_resolution_clock::time_point TimeStart;
};

void RegisterDelay(NodeActionsMap&, std::set<flatbuffers::Type const*> const&);

void RegisterUtil(NodeActionsMap& functions, std::set<flatbuffers::Type const*> const& types)
{
	functions["mz.Time"].NodeCreated = [](auto const&, auto&, void** context) {
		*context = new TimeNodeContext();
	};
	functions["mz.Time"].EntryPoint = [](mz::Args& pins, void* ctx) {
		*pins.Get<float>("Seconds") = static_cast<TimeNodeContext*>(ctx)->GetDeltaTime();
		return true;
	};
	functions["mz.Time"].NodeRemoved = [](void* context, auto const&) {
		delete static_cast<TimeNodeContext*>(context);
	};
	functions["mz.CalculateNodalPoint"].EntryPoint = [](mz::Args& args, void* ctx){
		auto pos = args.Get<glm::dvec3>("Camera Position");
		auto rot = args.Get<glm::dvec3>("Camera Orientation");
		auto sca = args.Get<f64>("Nodal Offset");
		auto out = args.Get<glm::dvec3>("Nodal Point");
		glm::dvec2 ANG = glm::radians(glm::dvec2(rot->z, rot->y));
		glm::dvec2 COS = cos(ANG);
		glm::dvec2 SIN = sin(ANG);
		glm::dvec3 f = glm::dvec3(COS.y * COS.x, COS.y * SIN.x, SIN.y);
		*out = *pos + f **sca;
		return true;
	};
	RegisterImageIO(functions);
	RegisterDelay(functions, types);
	RegisterChannelViewer(functions);
	RegisterSwitcher(functions);
}

} // namespace mz