// Copyright MediaZ Teknoloji A.S. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <Nodos/Name.hpp>
#include <Nodos/Helpers.hpp>

#include "nosVariableSubsystem/nosVariableSubsystem.h"
#include "./EditorEvents_generated.h"

NOS_INIT_WITH_MIN_REQUIRED_MINOR(0); // APITransition: Reminder that this should be reset after next major!

NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

namespace nos::sys::variables
{
std::unordered_map<uint32_t, nosVariableSubsystem*> GExportedSubsystemVersions;

struct VariableInfo
{
	nos::Name Name;
	nos::Name TypeName;
	nos::Buffer Value;
	std::unordered_set<uuid> References;
};
	
struct VariableManager
{
	VariableManager(const VariableManager&) = delete;
	VariableManager& operator=(const VariableManager&) = delete;
	static VariableManager& GetInstance() { return Instance; }

	nosResult Set(nosName name, nosName typeName, const nosBuffer* inValue)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it == Variables.end())
		{
			nos::Name newName = name;
			nosEngine.LogI("Creating variable %s", newName.AsCStr());
			auto& variable = (Variables[newName] = VariableInfo{ newName, typeName, *inValue });
			OnVariableAdded(variable);
		}
		else
		{
			it->second.TypeName = typeName;
			it->second.Value = *inValue;
			OnVariableUpdated(it->second);
		}
		return NOS_RESULT_SUCCESS;
	}

	nosResult Get(nosName name, nosName* outTypeName, nosBuffer* outValue)
	{
		std::shared_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it == Variables.end())
			return NOS_RESULT_NOT_FOUND;
		*outTypeName = it->second.TypeName;
		*outValue = it->second.Value;
		return NOS_RESULT_SUCCESS;
	}

	nosResult AddNodeReference(nosName name, nosUUID nodeId)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it == Variables.end())
			return NOS_RESULT_NOT_FOUND;
		it->second.References.insert(nodeId);
		SendVariableReferencesToEditors(it->second);
		return NOS_RESULT_SUCCESS;
	}

	nosResult DeleteNodeReference(nosName name, nosUUID nodeId)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it == Variables.end())
			return NOS_RESULT_NOT_FOUND;
		auto& refs = it->second.References;
		auto refIt = refs.find(nodeId);
		if (refIt == refs.end())
			return NOS_RESULT_FAILED;
		refs.erase(refIt);
		SendVariableReferencesToEditors(it->second);
		return NOS_RESULT_SUCCESS;
	}

	nosResult DeleteVariable(nos::Name name)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = Variables.find(name);
		if (it == Variables.end())
			return NOS_RESULT_NOT_FOUND;
		if (!it->second.References.empty())
		{
			nosEngine.LogE("Cannot delete variable %s, it still has %d references", name.AsCStr(), it->second.References.size());
			return NOS_RESULT_FAILED;
		}
		nosEngine.LogI("Deleting variable %s", name.AsCStr());
		Variables.erase(it);
		OnVariableDeleted(name);
		return NOS_RESULT_SUCCESS;
	}

	int32_t RegisterVariableUpdateCallback(nosName name, nosVariableUpdateCallback callback, void* userData)
	{
		std::unique_lock lock(VariablesMutex);
		VariableUpdateCallbacks[name][++NextCallbackId] = { callback, userData };
		auto it = Variables.find(name);
		if (it != Variables.end())
			callback(name, userData, it->second.TypeName, it->second.Value.GetInternal());
		return NextCallbackId;
	}

	nosResult UnregisterVariableUpdateCallback(nosName name, int32_t callbackId)
	{
		std::unique_lock lock(VariablesMutex);
		auto it = VariableUpdateCallbacks.find(name);
		if (it == VariableUpdateCallbacks.end())
			return NOS_RESULT_NOT_FOUND;
		auto& callbacks = it->second;
		auto cbIt = callbacks.find(callbackId);
		if (cbIt == callbacks.end())
			return NOS_RESULT_NOT_FOUND;
		callbacks.erase(cbIt);
		return NOS_RESULT_SUCCESS;
	}

	void OnEditorConnected(uint64_t editorId)
	{
		SendVariableListToEditors(editorId);
		for (auto& [id, variable] : Variables)
			SendVariableReferencesToEditors(variable, editorId);
	}

private:
	VariableManager() = default;

protected:
	void OnVariableListUpdated()
	{
		SendVariableListToEditors();
		SendVariableNameStringListUpdate();
	}

	void OnVariableAdded(VariableInfo& variable)
	{
		SendVariableNameStringListUpdate();
		OnVariableUpdated(variable);
	}

	void OnVariableUpdated(VariableInfo& variable)
	{
		SendVariableToEditors(variable);
		SendVariableToListeners(variable);
	}

	void OnVariableDeleted(nos::Name name)
	{
		flatbuffers::FlatBufferBuilder fbb;
		auto offset= editor::CreateVariableDeletedDirect(fbb, name.AsCStr());
		auto event = editor::CreateFromSubsystem(fbb, editor::FromSubsystemUnion::VariableDeleted, offset.Union());
		fbb.Finish(event);
		nos::Buffer buf = fbb.Release();
		nosSendEditorMessageParams params{.Message = buf, .DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_BROADCAST};
		nosEngine.SendEditorMessage(&params);
	}

	void SendVariableListToEditors(std::optional<uint64_t> optEditorId = std::nullopt)
	{
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<Variable>> variables;
		for (auto& [id, props] : Variables)
		{
			std::vector<uint8_t> buf = props.Value;
			auto variableOffset = CreateVariableDirect(fbb, id.AsCStr(), props.TypeName.AsCStr(), &buf);
			variables.push_back(variableOffset);
		}
		auto offset = editor::CreateVariableListDirect(fbb, &variables);
		auto event  = editor::CreateFromSubsystem(fbb, editor::FromSubsystemUnion::VariableList, offset.Union());
		fbb.Finish(event);
		nos::Buffer buf = fbb.Release();
		nosSendEditorMessageParams params{.Message = buf};
		if (optEditorId)
		{
			params.DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_TO_SELECTED;
			params.ToSelected = {.EditorId = *optEditorId};
		}
		else
			params.DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_BROADCAST;
		nosEngine.SendEditorMessage(&params);
	}

	void SendVariableNameStringListUpdate()
	{
		std::vector<std::string> names;
		for (auto& [id, props] : Variables)
			names.push_back(id.AsString());
		UpdateStringList("nos.sys.variables.Names", names);
	}

	void SendVariableToEditors(VariableInfo& variable)
	{
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<uint8_t> buf = variable.Value;
		auto offset = CreateVariableDirect(fbb, variable.Name.AsCStr(), variable.TypeName.AsCStr(), &buf);
		auto event  = editor::CreateFromSubsystem(fbb, editor::FromSubsystemUnion::nos_sys_variables_Variable, offset.Union());
		fbb.Finish(event);
		nos::Buffer msgBuf = fbb.Release();
		nosSendEditorMessageParams params{.Message = msgBuf,
										  .DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_BROADCAST};
		nosEngine.SendEditorMessage(&params);
	}

	void SendVariableToListeners(VariableInfo& variable)
	{
		auto it = VariableUpdateCallbacks.find(variable.Name);
		if (it == VariableUpdateCallbacks.end())
			return;
		for (auto& [id, pr] : it->second)
		{
			auto& [callback, userData] = pr;
			callback(variable.Name, userData, variable.TypeName, variable.Value.GetInternal());
		}
	}

	void SendVariableReferencesToEditors(VariableInfo& variable, std::optional<uint64_t> optEditorId = std::nullopt)
	{
		flatbuffers::FlatBufferBuilder fbb;
		std::vector<fb::UUID> refs;
		for (auto& ref : variable.References)
			refs.push_back(ref);
		auto refsEvent = editor::CreateVariableReferencesDirect(fbb, variable.Name.AsCStr(), &refs);
		auto event = editor::CreateFromSubsystem(fbb, editor::FromSubsystemUnion::VariableReferences, refsEvent.Union());
		fbb.Finish(event);
		nos::Buffer buf = fbb.Release();
		nosSendEditorMessageParams params{ .Message = buf };
		if (optEditorId)
		{
			params.DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_TO_SELECTED;
			params.ToSelected = { .EditorId = *optEditorId };
		}
		else
			params.DispatchType = NOS_EDITOR_MESSAGE_DISPATCH_TYPE_BROADCAST;
		nosEngine.SendEditorMessage(&params);
	}

	static VariableManager Instance;

	std::shared_mutex VariablesMutex;
	std::unordered_map<nos::Name, VariableInfo> Variables;
	std::unordered_map<nos::Name, std::unordered_map<int32_t, std::pair<nosVariableUpdateCallback, void*>>> VariableUpdateCallbacks;
	int32_t NextCallbackId = 0;
};

VariableManager VariableManager::Instance{};

nosResult NOSAPI_CALL Set(nosName name, nosName typeName, const nosBuffer* inValue)
{
	if (!name.ID || !inValue)
		return NOS_RESULT_INVALID_ARGUMENT;
	return VariableManager::GetInstance().Set(name, typeName, inValue);
}

nosResult NOSAPI_CALL Get(nosName name, nosName* outTypeName, nosBuffer* outValue)
{
	if (!name.ID || !outValue)
		return NOS_RESULT_INVALID_ARGUMENT;
	return VariableManager::GetInstance().Get(name, outTypeName, outValue);
}

int32_t NOSAPI_CALL RegisterVariableUpdateCallback(nosName name, nosVariableUpdateCallback callback, void* userData)
{
	return VariableManager::GetInstance().RegisterVariableUpdateCallback(name, callback, userData);
}

nosResult NOSAPI_CALL UnregisterVariableUpdateCallback(nosName name, int32_t callbackId)
{
	return VariableManager::GetInstance().UnregisterVariableUpdateCallback(name, callbackId);
}

nosResult NOSAPI_CALL AddNodeReference(nosName name, nosUUID nodeId)
{
	return VariableManager::GetInstance().AddNodeReference(name, nodeId);
}

nosResult NOSAPI_CALL DeleteNodeReference(nosName name, nosUUID nodeId)
{
	return VariableManager::GetInstance().DeleteNodeReference(name, nodeId);
}
	
nosResult NOSAPI_CALL Export(uint32_t minorVersion, void** outSubsystemContext)
{
	auto it = GExportedSubsystemVersions.find(minorVersion);
	if (it != GExportedSubsystemVersions.end())
	{
		*outSubsystemContext = it->second;
		return NOS_RESULT_SUCCESS;
	}
	auto* subsystem = new nosVariableSubsystem();
	subsystem->Get = Get;
	subsystem->Set = Set;
	subsystem->RegisterVariableUpdateCallback = RegisterVariableUpdateCallback;
	subsystem->UnregisterVariableUpdateCallback = UnregisterVariableUpdateCallback;
	subsystem->AddNodeReference = AddNodeReference;
	subsystem->DeleteNodeReference = DeleteNodeReference;
	*outSubsystemContext = subsystem;
	GExportedSubsystemVersions[minorVersion] = subsystem;
	return NOS_RESULT_SUCCESS;
}

nosResult NOSAPI_CALL Initialize()
{
	return NOS_RESULT_SUCCESS;
}

nosResult NOSAPI_CALL UnloadSubsystem()
{
	return NOS_RESULT_SUCCESS;
}

namespace editor
{
void NOSAPI_CALL OnMessageFromEditor(uint64_t editorId, nosBuffer message)
{
	auto msg = flatbuffers::GetRoot<FromEditor>(message.Data);
	switch (msg->event_type())
	{
	case FromEditorUnion::SetVariable:
	{
		auto setVar = msg->event_as_SetVariable();
		if (auto variable = setVar->variable())
		{
			nos::Name name(variable->name()->c_str());
			nos::Name typeName(variable->type_name()->c_str());
			nosBuffer value{(void*)variable->value()->data(), variable->value()->size()};
			VariableManager::GetInstance().Set(name, typeName, &value);
		}
		break;
	}
	case FromEditorUnion::DeleteVariable:
	{
		auto delVar = msg->event_as_DeleteVariable();
		if (auto name = delVar->name())
		{
			nos::Name varName(name->c_str());
			VariableManager::GetInstance().DeleteVariable(varName);
		}
		break;
	}
	default:
		break;
	}
}
}

extern "C"
{
NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
{
	subsystemFunctions->OnRequest = Export;
	subsystemFunctions->Initialize = Initialize;
	subsystemFunctions->OnPreUnloadSubsystem = UnloadSubsystem;
	subsystemFunctions->OnEditorConnected = [](uint64_t editorId)
	{
		VariableManager::GetInstance().OnEditorConnected(editorId);
	};
	subsystemFunctions->OnMessageFromEditor = editor::OnMessageFromEditor;
	return NOS_RESULT_SUCCESS;
}
}
}
