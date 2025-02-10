// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include <tinyexpr.h>
#include <random>

namespace nos::math
{
struct RandomNode : NodeContext
{
	RandomNode(nosFbNodePtr node) : NodeContext(node)
	{
		AddPinValueWatcher(NOS_NAME("Seed"), [this](nos::Buffer const& newVal, auto const& oldVal) {
			auto& seed = *newVal.As<int>();
			Generator.seed(seed);
		});
		AddPinValueWatcher(NOS_NAME("Min"), [this](nos::Buffer const& newVal, auto const& oldVal) {
			auto& minVal = *newVal.As<float>();
			Dist = std::uniform_real_distribution(minVal, Dist.b());
		});
		AddPinValueWatcher(NOS_NAME("Max"), [this](nos::Buffer const& newVal, auto const& oldVal) {
			auto& maxVal = *newVal.As<float>();
			Dist = std::uniform_real_distribution(Dist.a(), maxVal);
		});
	}	

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nosEngine.SetPinValueByName(NodeId, NOS_NAME("Random"), nos::Buffer::From(Dist(Generator)));
		return NOS_RESULT_SUCCESS;
	}

	std::mt19937 Generator;
	std::uniform_real_distribution<float> Dist;
};

void RegisterRandom(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Random"), RandomNode, fn);
}
} // namespace nos::math

