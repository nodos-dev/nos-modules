// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

namespace nos::strings
{
struct Json2PinNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		NodeExecuteParams execParams(params);
		auto& jsonPin = execParams[NOS_NAME("Json")];
		auto& outPin = execParams[NOS_NAME("Out")];
		if(outPin.TypeName == NOS_NAME("nos.fb.Void"))
		{
			SetNodeStatusMessage("Out pin is not connected to typed pin", fb::NodeStatusMessageType::FAILURE);
			return NOS_RESULT_FAILED;
		}
		nosBuffer outBuffer;
		auto ret = nosEngine.GenerateBufferFromJson(outPin.TypeName, (const char*)(*jsonPin.Data).Data, &outBuffer);
		if (ret == NOS_RESULT_SUCCESS)
		{
			SetPinValue(outPin.Name, outBuffer);
			ClearNodeStatusMessages();
		}
		else
		{
			SetNodeStatusMessage("Unable to convert data to JSON", fb::NodeStatusMessageType::FAILURE);
		}
		return NOS_RESULT_SUCCESS;
	}

	void OnPinDisconnected(nos::Name pinName) override
	{
		if (pinName == NOS_NAME("Out"))
		{
			SetPinType(NOS_NAME("Out"), NOS_NAME("nos.fb.Void"));
		}
	}
};


nosResult RegisterJson2Pin(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Json2Pin"), Json2PinNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos