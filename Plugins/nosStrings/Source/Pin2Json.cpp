// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

namespace nos::strings
{
struct Pin2JsonNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		NodeExecuteParams execParams(params);
		auto& dataPin = execParams[NOS_NAME("Data")];
		auto& jsonPin = execParams[NOS_NAME("Json")];
		if (auto outJson = GenerateJsonFromBuffer(dataPin.TypeName, *dataPin.Data))
		{
			SetPinValue(jsonPin.Name, outJson->AsBuffer());
			ClearNodeStatusMessages();
		}
		else
			SetNodeStatusMessage("Unable to convert data to JSON", fb::NodeStatusMessageType::FAILURE);
		return NOS_RESULT_SUCCESS;
	}

	void OnPinDisconnected(nos::Name pinName) override
	{
		if (pinName == NOS_NAME("Data"))
		{
			SetPinType(NOS_NAME("Data"), NOS_NAME("nos.Generic"));
		}
	}
};


nosResult RegisterPin2Json(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Pin2Json"), Pin2JsonNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos