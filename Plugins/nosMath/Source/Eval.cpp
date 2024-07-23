// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include <tinyexpr.h>

namespace nos::math
{
struct EvalNodeContext : NodeContext
{
	enum MenuCommand
	{
		ADD_INPUT = 1
	};
	EvalNodeContext(nosFbNode const* node) : NodeContext(node)
	{
		Resolve(node);
	}

	void OnNodeUpdated(const fb::Node* updatedNode) override
	{
		Resolve(updatedNode);
	}

	void Resolve(const nosFbNode* node)
	{
		auto pinCount = node->pins()->size();
		Variables.clear();
		for (auto i = 0; i < pinCount; i++)
		{
			auto pin = node->pins()->Get(i);
			if (pin->show_as() == fb::ShowAs::INPUT_PIN)
			{
				// If pin name starts with "In" then use its display name as variable name.
				if (strncmp(pin->name()->c_str(), "In", 2) == 0)
				{
					auto uniqueName = pin->name()->str();
					Variables.insert(uniqueName);
				}
			}
		}
	}
	
	void OnNodeMenuRequested(const nosContextMenuRequest* request) override
	{
		flatbuffers::FlatBufferBuilder fbb;
		std::vector items = {
			nos::CreateContextMenuItemDirect(fbb, "Add Input", ADD_INPUT, nullptr)
		};
		HandleEvent(CreateAppEvent(fbb, app::CreateAppContextMenuUpdateDirect(
			                           fbb, request->item_id(), request->pos(), request->instigator(),
			                           &items
		                           )));
	}
	
	void OnMenuCommand(nosUUID itemID, uint32_t cmd) override
	{
		if (cmd != ADD_INPUT)
			return;
		flatbuffers::FlatBufferBuilder fbb;
		nosUUID pinId{};
		nosEngine.GenerateID(&pinId);
		std::string pinName = "In_" + std::to_string(Variables.size());
		std::vector pins = {
			fb::CreatePinDirect(fbb, &pinId, pinName.c_str(), "double", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_OR_PROPERTY)
		};
		HandleEvent(CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, &pins)));
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs pins(args);

		std::set<te_variable> vars;
		for (auto const& uniqueName : Variables)
		{
			te_variable var(uniqueName.c_str(), (double*)pins[nos::Name(uniqueName)].Data->Data);
			vars.insert(std::move(var));
		}
		Parser.set_variables_and_functions(vars);
		auto exprStr = InterpretPinValue<const char>(pins[NOS_NAME("Expression")].Data->Data);
		if (strlen(exprStr) == 0)
		{
			SetNodeStatusMessage("No math expression is provided", fb::NodeStatusMessageType::WARNING);
			return NOS_RESULT_SUCCESS;
		}

		auto res = Parser.evaluate(exprStr);
		if (!Parser.success())
		{
			SetNodeStatusMessage("Error in expression: " + std::string(exprStr), fb::NodeStatusMessageType::FAILURE);
			return NOS_RESULT_FAILED;
		}
		SetNodeStatusMessage(std::string(exprStr) + " = " + std::to_string(res), fb::NodeStatusMessageType::INFO);
		SetPinValue(NOS_NAME("Result"), nos::Buffer::From(res));
		return NOS_RESULT_SUCCESS;
	}

	te_parser Parser;
	std::unordered_set<std::string> Variables; // unique name to display name
};

void RegisterEval(nosNodeFunctions* fn)
{
    NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Eval"), EvalNodeContext, fn);
}
} // namespace nos::math

