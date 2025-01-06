// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>
#include <tinyexpr.h>
#include <list>

namespace nos::math
{
struct EvalNodeContext : NodeContext
{
	enum MenuCommandType : uint8_t
	{
		ADD_INPUT = 0,
		REMOVE_INPUT = 1,
	};

	struct MenuCommand
	{
		MenuCommandType Type;
		uint8_t InputIndex;
		MenuCommand(uint32_t cmd) {
			Type = static_cast<MenuCommandType>(cmd & 0xFF);
			InputIndex = static_cast<uint8_t>((cmd >> 8) & 0xFF);
 		}
		MenuCommand(MenuCommandType type, uint8_t inputIndex) : Type(type), InputIndex(inputIndex) {}
		operator uint32_t() const
		{
			return (InputIndex << 8) | Type;
		}
	};

	std::vector<nosUUID> Inputs;
	
	EvalNodeContext(nosFbNode const* node) : NodeContext(node)
	{
		auto pinCount = node->pins()->size();
		std::list<nosUUID> pinsToUnorphan;
		for (auto i = 2; i < pinCount; i++)
		{
			auto pin = node->pins()->Get(i);
			if (pin->show_as() == fb::ShowAs::INPUT_PIN
				// Flag pins
				&& pin->name()->string_view() != "Show_Expression")
			{
				Variables[*pin->id()] = 0.0;
				Inputs.push_back(*pin->id());
				if (auto orphanState = pin->orphan_state()) {
					if (orphanState->type() == fb::PinOrphanStateType::ORPHAN)
						pinsToUnorphan.push_back(*pin->id());
				}
			}
		}
		Compile();
		for (auto const& pinId : pinsToUnorphan)
			SetPinOrphanState(pinId, fb::PinOrphanStateType::ACTIVE);
	}

	void OnPartialNodeUpdated(nosNodeUpdate const* update) override
	{
		if (update->Type == NOS_NODE_UPDATE_PIN_DELETED)
		{
			// Since pin is deleted from the NodeContext's pins map, we can't get its name from its id.
			Variables.erase(update->PinDeleted);
			std::erase_if(Inputs, [&](auto id) {return id == update->PinDeleted; });
			Compile();
		}
		else if (update->Type == NOS_NODE_UPDATE_PIN_CREATED)
		{
			auto* pin = update->PinCreated;
			if (pin->show_as() == fb::ShowAs::INPUT_PIN)
			{
				Variables.try_emplace(*pin->id(), 0.0);
				Inputs.push_back(*pin->id());
				Compile();
			}
		}
	}

	void OnPinValueChanged(nos::Name pinName, nosUUID pinId, nosBuffer value) override
	{
		if (pinName == NOS_NAME("Show_Expression"))
		{
			auto newVal = *InterpretPinValue<bool>(value.Data);
			if (ShowExpressionInNode != newVal)
			{
				ShowExpressionInNode = newVal;
				Compile();
			}
		}
		else if (pinName == NOS_NAME("Expression"))
		{
			auto exprStr = InterpretPinValue<const char>(value.Data);
			if (strlen(exprStr) == 0)
				SetStatus("No math expression is provided", fb::NodeStatusMessageType::WARNING);
			nosEngine.LogI("Compiling expression: %s", exprStr);
			if (strcmp(Expression.c_str(), exprStr) != 0)
			{
				Expression = exprStr;
				Compile();
			}
		}
	}

	bool ShowExpressionInNode = false;

	void SetStatus(const std::string& message, fb::NodeStatusMessageType type)
	{
		nosEngine.LogD("Eval: Setting node status");
		if (type == fb::NodeStatusMessageType::FAILURE)
		{
			SetPinOrphanState(NOS_NAME("Result"), fb::PinOrphanStateType::ORPHAN, message.c_str());
			SetNodeStatusMessage(message, type);
		}
		else
		{
			SetPinOrphanState(NOS_NAME("Result"), fb::PinOrphanStateType::ACTIVE);
			if (ShowExpressionInNode)
				SetNodeStatusMessage(message, type);
			else
				ClearNodeStatusMessages();
		}
		Status = { message, type };
	}

	void OnPinUpdated(const nosPinUpdate* update) override {
		if (update->UpdatedField != NOS_PIN_FIELD_DISPLAY_NAME)
			return;
		std::string newDisplayName = nos::Name(update->DisplayName).AsString();
		ClearNodeStatusMessages();
		SetPinOrphanState(NOS_NAME("Result"), fb::PinOrphanStateType::ACTIVE);
		Compile();
	}
	
	void OnNodeMenuRequested(const nosContextMenuRequest* request) override
	{
		uint32_t cmd = MenuCommand(ADD_INPUT, 0);
		
		flatbuffers::FlatBufferBuilder fbb;
		std::vector items = {
			nos::CreateContextMenuItemDirect(fbb, "Add Input", cmd, nullptr)
		};
		HandleEvent(CreateAppEvent(fbb, app::CreateAppContextMenuUpdateDirect(
			                           fbb, request->item_id(), request->pos(), request->instigator(),
			                           &items
		                           )));
	}

	void OnPinMenuRequested(nos::Name pinName, const nosContextMenuRequest* request) override
	{
		flatbuffers::FlatBufferBuilder fbb;
		if (pinName == NOS_NAME("Result") || pinName == NOS_NAME("Show_Expression") || pinName == NOS_NAME("Expression"))
			return;
		auto index = std::distance(Inputs.begin(), std::find(Inputs.begin(), Inputs.end(), *GetPinId(pinName)));
		uint32_t cmd = MenuCommand(REMOVE_INPUT, index);
		std::vector items = {
			nos::CreateContextMenuItemDirect(fbb, "Remove Input", cmd, nullptr)
		};
		HandleEvent(CreateAppEvent(fbb, app::CreateAppContextMenuUpdateDirect(
			                           fbb, request->item_id(), request->pos(), request->instigator(),
			                           &items
		                           )));
	}
	
	void OnMenuCommand(nosUUID itemID, uint32_t cmd) override
	{
		auto command = MenuCommand(cmd);

		switch (command.Type)
		{
		case ADD_INPUT:
		{
			flatbuffers::FlatBufferBuilder fbb;
			nosUUID pinId = nosEngine.GenerateID();
			constexpr std::string_view VARIABLE_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
			if (Variables.size() >= VARIABLE_NAMES.size())
			{
				SetStatus("Maximum number of inputs reached", fb::NodeStatusMessageType::WARNING);
				return;
			}
			// Find the first available variable name
			
			std::unordered_set<char> variableNames;
			for (auto const& [pinId, value] : Variables)
				variableNames.insert(*GetPinName(pinId)->AsCStr());
			std::string pinName;
			for (size_t i = 0; i < VARIABLE_NAMES.size(); i++)
			{
				if (variableNames.find(VARIABLE_NAMES[i]) == variableNames.end())
				{
					pinName = VARIABLE_NAMES[i];
					break;
				}
			}
			if (pinName.empty())
			{
				SetStatus("Failed to add input", fb::NodeStatusMessageType::FAILURE);
				return;
			}
			std::vector pins = {
				fb::CreatePinDirect(fbb, &pinId, pinName.c_str(), "double", fb::ShowAs::INPUT_PIN, fb::CanShowAs::INPUT_PIN_OR_PROPERTY)
			};
			HandleEvent(CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, 0, &pins)));
			break;
		}
		case REMOVE_INPUT:
		{
			auto pinId = Inputs[command.InputIndex];
			flatbuffers::FlatBufferBuilder fbb;
			std::vector pinsToRemove {
				fb::UUID(pinId),
			};
			HandleEvent(CreateAppEvent(fbb, CreatePartialNodeUpdateDirect(fbb, &NodeId, ClearFlags::NONE, &pinsToRemove)));
			break;
		}
		}
	}

	bool Compile()
	{
		if (Expression.empty())
			return true;
		std::set<te_variable> vars;
		std::unordered_set<nos::Name> displayNames;
		for (auto const& [uniqueName, value] : Variables)
		{
			auto* pin = GetPin(uniqueName);
			te_variable var(pin->DisplayName.AsCStr(), &value);
			vars.insert(std::move(var));
			if (!displayNames.insert(pin->DisplayName).second)
			{
				SetStatus("Duplicate name: " + pin->DisplayName.AsString(), fb::NodeStatusMessageType::FAILURE);
				return false;
			}
		}
		try
		{
			Parser.set_variables_and_functions(vars);
			if (!Parser.compile(Expression.c_str()))
			{
				SetStatus("Failed to compile expression", fb::NodeStatusMessageType::FAILURE);
				return false;
			}
		} catch (std::runtime_error& err) {
			SetStatus(err.what(), fb::NodeStatusMessageType::FAILURE);
			return false;
		}
		SetStatus(Expression, fb::NodeStatusMessageType::INFO);
		return true;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nos::NodeExecuteParams pins(params);
		
		for (auto const& [pinId, value] : Variables)
			Variables[pinId] = *InterpretPinValue<double>(pins[*GetPinName(pinId)].Data->Data);

		try
		{
			auto res = Parser.evaluate();
			SetPinValue(NOS_NAME("Result"), nos::Buffer::From(res));
			return NOS_RESULT_SUCCESS;
		}
		catch (std::runtime_error& err)
		{
			SetStatus(err.what(), fb::NodeStatusMessageType::FAILURE);
			return NOS_RESULT_FAILED;
		}
	}

	te_parser Parser;
	std::string Expression;
	std::unordered_map<nosUUID, double> Variables;

	struct {
		std::string Message;
		fb::NodeStatusMessageType Type;
	} Status = {};
};

void RegisterEval(nosNodeFunctions* fn)
{
    NOS_BIND_NODE_CLASS(NOS_NAME("nos.math.Eval"), EvalNodeContext, fn);
}
} // namespace nos::math

