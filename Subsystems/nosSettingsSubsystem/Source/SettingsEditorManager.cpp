// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>
#include "Globals.h"
#include "SettingsEditorManager.h"
#include "SettingsFileManager.h"

namespace nos::sys::settings
{
nosResult SettingsEditorManager::RegisterEditorSettings(u64 itemCount, const nosSettingsEditorItem** itemList, nosPfnSettingsItemUpdate itemUpdateCallback, nosSettingsFileDirectory saveDirectory)
{
	nosModuleInfo module = {};
	if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
		return NOS_RESULT_FAILED;

	if (itemUpdateCallback == nullptr) {
		nosEngine.LogE("Module %s sent a null callback for RegisterEditorSettings", nos::Name(module.Id.Name).AsCStr());
		return NOS_RESULT_FAILED;
	}

	nos::sys::settings::editor::TSettingsUpdateFromSubsystem update;
	update.module_name = nos::Name(module.Id.Name);
	update.settings.clear();
	for (u64 i = 0; i < itemCount; i++) {
		auto& item = itemList[i];
		auto& newItem = update.settings.emplace_back();
		newItem.reset(new editor::TSettingsEditorItem());
		item->UnPackTo(newItem.get());
	}

	std::unique_lock lock(RegisteredModulesMutex);
	auto& registeredModuleInfo = RegisteredModules[nos::Name(module.Id.Name).AsString()];
	registeredModuleInfo.ItemUpdateCallback = itemUpdateCallback;
	registeredModuleInfo.ModuleInfo = module;
	registeredModuleInfo.UpdateToEditor = nos::Buffer::From(update);
	registeredModuleInfo.SaveDirectory = saveDirectory;
	nosEngine.SendCustomMessageToEditors(NOS_NAME_STATIC(NOS_SETTINGS_SUBSYSTEM_NAME), registeredModuleInfo.UpdateToEditor);
	return NOS_RESULT_SUCCESS;
}

nosResult SettingsEditorManager::UnregisterEditorSettings()
{
	nosModuleInfo module = {};
	if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
		return NOS_RESULT_FAILED;
	std::unique_lock lock(RegisteredModulesMutex);
	auto moduleName = nos::Name(module.Id.Name);
	if (auto it = RegisteredModules.find(moduleName.AsString()); it != RegisteredModules.end())
	{
		nos::sys::settings::editor::TSettingsUpdateFromSubsystem list;
		list.module_name = moduleName;
		nosEngine.SendCustomMessageToEditors(NOS_NAME_STATIC(NOS_SETTINGS_SUBSYSTEM_NAME), nos::Buffer::From(list));
		RegisteredModules.erase(it);
		return NOS_RESULT_SUCCESS;
	}
	nosEngine.LogE("Module %s is not registered for editor settings, but called UnregisterEditorSettings().", nosEngine.GetString(module.Id.Name));
	return NOS_RESULT_FAILED;
}

void SettingsEditorManager::OnEditorConnected(uint64_t editorId)
{
	for (auto& [moduleName, registeredModuleInfo] : RegisteredModules) {
		nosEngine.SendCustomMessageToEditors(NOS_NAME_STATIC(NOS_SETTINGS_SUBSYSTEM_NAME), registeredModuleInfo.UpdateToEditor);
	}
}

void SettingsEditorManager::OnMessageFromEditor(uint64_t editorId, nosBuffer message)
{
	auto& regModules = RegisteredModules;
	auto msg = flatbuffers::GetMutableRoot<nos::sys::settings::editor::SettingsUpdateFromEditor>(message.Data);
	auto moduleName = msg->module_name()->c_str();
	auto it = regModules.find(moduleName);
	if (it == regModules.end()) {
		nosEngine.LogE("Module %s is not registered for editor settings, but Editor sent a message.", moduleName);
		return;
	}

	auto ret = it->second.ItemUpdateCallback(msg->entry()->entry_name()->c_str(), *nos::Buffer(msg->entry()->data()).GetInternal());
	if (ret == NOS_RESULT_SUCCESS) {
		nosSettingsEntryParams params;
		params.Directory = it->second.SaveDirectory;
		params.EntryName = msg->entry()->entry_name()->c_str();
		params.TypeName = nos::Name(msg->entry()->type_name()->c_str());
		params.Buffer = { msg->mutable_entry()->mutable_data()->data(), msg->entry()->data()->size() };
		GSettingsFileManager->WriteSettings(&params, it->second.ModuleInfo);
	}
}
}