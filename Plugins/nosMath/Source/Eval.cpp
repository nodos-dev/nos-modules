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
		auto prevDisplayNames = std::move(DisplayNames);
		decltype(Variables) newVariables;
		for (auto i = 0; i < pinCount; i++)
		{
			auto pin = node->pins()->Get(i);
			if (pin->show_as() == fb::ShowAs::INPUT_PIN)
			{
				if (strncmp(pin->name()->c_str(), "In", 2) == 0)
				{
					auto uniqueName = pin->name()->str();
					auto displayName = pin->display_name() ? pin->display_name()->str() : "In" + uniqueName.substr(2);
					auto value = InterpretPinValue<double>((void*)pin->data()->Data());
					newVariables[nos::Name(uniqueName)] = *value;
					DisplayNames[nos::Name(uniqueName)] = displayName;
				}
			}
		}
		for (auto const& [uniqueName, value] : newVariables)
		{
			Variables.try_emplace(uniqueName, value);
		}
		for (auto it = Variables.begin(); it != Variables.end();)
		{
			if (newVariables.find(it->first) == newVariables.end())
				it = Variables.erase(it);
			else
				++it;
		}
		if (prevDisplayNames != DisplayNames)
			Compile();
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

	bool Compile()
	{
		if (Expression.empty())
			return true;
		std::set<te_variable> vars;
		std::unordered_set<std::string> displayNames;
		for (auto const& [uniqueName, value] : Variables)
		{
			auto displayName = DisplayNames[uniqueName];
			te_variable var(displayName.c_str(), &value);
			vars.insert(std::move(var));
			if (!displayNames.insert(displayName).second)
			{
				SetNodeStatusMessage("Duplicate name: " + displayName, fb::NodeStatusMessageType::FAILURE);
				return false;
			}
		}
		try
		{
			Parser.set_variables_and_functions(vars);
			if (!Parser.compile(Expression.c_str()))
			{
				SetNodeStatusMessage("Failed to compile expression", fb::NodeStatusMessageType::FAILURE);
				return false;
			}
		} catch (std::runtime_error& err) {
			SetNodeStatusMessage(err.what(), fb::NodeStatusMessageType::FAILURE);
			return false;
		}
		SetNodeStatusMessage(Expression, fb::NodeStatusMessageType::INFO);
		return true;
	}

	nosResult ExecuteNode(const nosNodeExecuteArgs* args) override
	{
		nos::NodeExecuteArgs pins(args);

		auto exprStr = InterpretPinValue<const char>(pins[NOS_NAME("Expression")].Data->Data);
		if (strlen(exprStr) == 0)
		{
			SetNodeStatusMessage("No math expression is provided", fb::NodeStatusMessageType::WARNING);
			return NOS_RESULT_SUCCESS;
		}

		if (strcmp(Expression.c_str(), exprStr) != 0)
		{
			Expression = exprStr;
			if (!Compile())
				return NOS_RESULT_FAILED;
		}

		for (auto const& [uniqueName, value] : Variables)
		{
			auto pinValue = *InterpretPinValue<double>(pins[uniqueName].Data->Data);
			Variables[uniqueName] = pinValue;
		}

		try
		{
			auto res = Parser.evaluate();
			SetPinValue(NOS_NAME("Result"), nos::Buffer::From(res));
			return NOS_RESULT_SUCCESS;
		}
		catch (std::runtime_error& err)
		{
			SetNodeStatusMessage(err.what(), fb::NodeStatusMessageType::FAILURE);
			return NOS_RESULT_FAILED;
		}
	}

	te_parser Parser;
	std::string Expression;
	std::unordered_map<nos::Name, double> Variables;
	std::unordered_map<nos::Name, std::string> DisplayNames;
};

void RegisterEval(nosNodeFunctions* fn)
{
    NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Eval"), EvalNodeContext, fn);
}
} // namespace nos::math

