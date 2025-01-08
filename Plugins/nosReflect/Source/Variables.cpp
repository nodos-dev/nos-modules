#include <Nodos/PluginHelpers.hpp>

#include <shared_mutex>

#include "Names.h"

namespace nos::reflect
{
NOS_REGISTER_NAME(SetVariable)
NOS_REGISTER_NAME(GetVariable)

struct VariableContainer
{
	enum class UserType
	{
		Broadcaster,
		Listener,
	};
	
	void Associate(std::string const& name, nosUUID const& pinId, UserType type)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it != Variables.end())
			it->second.Users.insert({pinId, type});
		else
		{
			nosEngine.LogI("Creating variable %s", name.c_str());
			Variables[name] = Variable{};
			Variables[name].Users.insert({pinId, type});
			UpdateStrings(lock);
		}
	}

	void Disassociate(std::string const& name, nosUUID const& pinId)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it != Variables.end())
		{
			auto& variable = it->second;
			variable.Users.erase(pinId);
			if (variable.Users.empty())
			{
				nosEngine.LogI("Erasing variable %s", it->first.c_str());
				Variables.erase(it);
				UpdateStrings(lock);
			}
		}
	}

	void SetVariable(std::string const& name, nos::Buffer const& value)
	{
		std::unique_lock lock(VariablesMutex);
		bool update = !Variables.contains(name);
		auto& variable = Variables[name];
		variable.Value = value;
		if (update)
			UpdateStrings(lock);
		for (auto const& [user, type] : variable.Users)
		{
			if (type == UserType::Listener)
				nosEngine.SetPinValue(user, value);
		}
	}

	bool GetVariable(std::string const& name, nos::Buffer& outCopy)
	{
		std::shared_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it != Variables.end())
		{
			outCopy = nos::Buffer(it->second.Value);
			return true;
		}
		return false;
	}

	void SetType(std::string const& name, nos::Name const& typeName)
	{
		std::unique_lock lock(VariablesMutex);
		Variables[name].TypeName = typeName;
		NotifyUsersAboutTypeChange(name, typeName);
	}

	std::optional<nos::Name> GetType(std::string const& name)
	{
		std::shared_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it != Variables.end())
			return it->second.TypeName;
		return std::nullopt;
	}

	void UpdateStrings()
	{
		std::unique_lock lock(VariablesMutex);
		UpdateStrings(lock);
	}
	
protected:
	void NotifyUsersAboutTypeChange(std::string const& name, nos::Name const& typeName)
	{
		for (auto const& [user, listener] : Variables[name].Users)
		{
			NodeContext::SetPinType(user, typeName);
		}
	}

	void UpdateStrings(std::unique_lock<std::shared_mutex> &lock)
	{
		auto names = GetAllVariableNames();
		UpdateStringList("nos.reflect.VariableNames", names);
		std::stringstream s;
		s << "List of variables:\n";
		for (auto& name : names)
			s << name << '\n';
		nosEngine.LogDI(s.str().c_str(), "Variable list updated");
	}

	std::vector<std::string> GetAllVariableNames()
	{
		std::vector<std::string> names;
		names.reserve(Variables.size());
		for (auto const& [name, value] : Variables)
		{
			names.push_back(name);
		}
		return names;
	}

	struct Variable {
		nos::Buffer Value;
		std::optional<nos::Name> TypeName;
		std::unordered_map<nosUUID, UserType> Users;
	};
	std::unordered_map<std::string, Variable> Variables;
	std::shared_mutex VariablesMutex;
	
} GVariables = {};


enum class SetVariableStatusItem
{
	VariableName,
	TypeName,
};

struct VariableNodeBase : NodeContext
{
	VariableNodeBase(const nosFbNode* node) : NodeContext(node)
	{
		ValuePinId = nosUUID(*GetPinId(NOS_NAME("Value")));
		GVariables.UpdateStrings();
	}

	~VariableNodeBase() override
	{
		GVariables.Disassociate(Name, ValuePinId);
	}

	void CheckType()
	{
		auto valuePin = GetPin(NOS_NAME("Value"));
		if (auto existingType = GVariables.GetType(Name))
			SetPinType(ValuePinId, existingType.value());
		else if (valuePin->TypeName != NSN_VOID)
			GVariables.SetType(Name, valuePin->TypeName);
		else
			SetStatus(SetVariableStatusItem::TypeName, fb::NodeStatusMessageType::WARNING, "Type not set");
	}

	void UpdateStatus()
	{
		std::vector<fb::TNodeStatusMessage> messages;
		for (auto& [type, message] : StatusMessages)
			messages.push_back(message);
		SetNodeStatusMessages(messages);
	}

	void  SetStatus(SetVariableStatusItem item, fb::NodeStatusMessageType msgType, std::string text)
	{
		StatusMessages[item] = fb::TNodeStatusMessage{{}, std::move(text), msgType};
		UpdateStatus();
	}

	void ClearStatus(SetVariableStatusItem item)
	{
		StatusMessages.erase(item);
		UpdateStatus();
	}
	
	std::unordered_map<SetVariableStatusItem,  fb::TNodeStatusMessage> StatusMessages;
	nosUUID ValuePinId;
	std::string Name;
};
	
struct SetVariableNode : VariableNodeBase
{
	SetVariableNode(const nosFbNode* node) : VariableNodeBase(node)
	{
		// For editor to show changes without a scheduled node, we use pin value change callbacks.
		// Once we support this in the engine, we can move these to ExecuteNode function.
		AddPinValueWatcher(NOS_NAME("Name"), [this](const nos::Buffer& value,  std::optional<nos::Buffer> oldValue)
		{
			if (oldValue)
			{
				std::string oldName = static_cast<const char*>(oldValue->Data());
				GVariables.Disassociate(oldName, ValuePinId);
			}
			std::string newName = static_cast<const char*>(value.Data());
			if (newName.empty() && oldValue)
			{
				SetPinValue(NOS_NAME("Name"), *oldValue);
				return;
			}
			if (newName.empty())
			{
				SetStatus(SetVariableStatusItem::VariableName, fb::NodeStatusMessageType::WARNING, "Provide a name");
				return;
			}
			Name = std::move(newName);
			GVariables.Associate(Name, ValuePinId, VariableContainer::UserType::Broadcaster);
			if (Value)
				GVariables.SetVariable(Name, *Value);
			CheckType();
			SetStatus(SetVariableStatusItem::VariableName, fb::NodeStatusMessageType::INFO, Name);
		});
		AddPinValueWatcher(NOS_NAME("Value"), [this](const nos::Buffer& value,  std::optional<nos::Buffer> oldValue)
		{
			Value = value;
			if (Name.empty())
				return;
			GVariables.SetVariable(Name, value);
		});
	}

	void OnPinUpdated(const nosPinUpdate* pinUpdate) override
	{
		if (pinUpdate->UpdatedField != NOS_PIN_FIELD_TYPE_NAME)
			return;
		ClearStatus(SetVariableStatusItem::TypeName);
		if (Name.empty())
			return;
		if (TypeName && *TypeName == pinUpdate->TypeName)
			return;
		TypeName = pinUpdate->TypeName;
		if (auto existingType = GVariables.GetType(Name))
		{
			if (existingType.value() != pinUpdate->TypeName)
				SetPinType(ValuePinId, existingType.value());
		}
		else
		{
			GVariables.SetType(Name, pinUpdate->TypeName);
		}
	}

	std::optional<nos::Buffer> Value;
	std::optional<nos::Name> TypeName;

};


struct GetVariableNode : VariableNodeBase
{
	GetVariableNode(const nosFbNode* node) : VariableNodeBase(node)
	{
		AddPinValueWatcher(NOS_NAME("Name"), [this](const nos::Buffer& value,  std::optional<nos::Buffer> oldValue)
		{
			if (oldValue)
			{
				std::string oldName = static_cast<const char*>(oldValue->Data());
				GVariables.Disassociate(oldName, ValuePinId);
			}
			std::string newName = static_cast<const char*>(value.Data());
			if (newName.empty())
			{
				SetStatus(SetVariableStatusItem::VariableName, fb::NodeStatusMessageType::WARNING, "Provide a name");
				return;
			}
			if (newName == Name)
				return;
			Name = std::move(newName);
			nos::Buffer outCopy;
			if (!GVariables.GetVariable(Name, outCopy))
			{
				GVariables.UpdateStrings();
				SetPinValue(NOS_NAME("Name"), "");
				return;
			}
			GVariables.Associate(Name, ValuePinId, VariableContainer::UserType::Listener);
			SetNodeStatusMessage(Name, fb::NodeStatusMessageType::INFO);
			CheckType();
			SetPinValue(NOS_NAME("Value"), outCopy);
		});
	}
};
	
nosResult RegisterSetVariable(nosNodeFunctions* node)
{
	NOS_BIND_NODE_CLASS(NSN_SetVariable, SetVariableNode, node);
	return NOS_RESULT_SUCCESS;
}

nosResult RegisterGetVariable(nosNodeFunctions* node)
{
	NOS_BIND_NODE_CLASS(NSN_GetVariable, GetVariableNode, node);
	return NOS_RESULT_SUCCESS;
}
}
