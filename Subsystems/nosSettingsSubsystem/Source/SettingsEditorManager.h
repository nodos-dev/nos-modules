// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>

namespace nos::sys::settings
{
struct SettingsEditorManager
{
	struct RegisteredModuleSettings
	{
		nosModuleInfo ModuleInfo;
		nos::Buffer UpdateToEditor;
		nosPfnSettingsItemUpdate ItemUpdateCallback = nullptr;
		nosSettingsFileDirectory SaveDirectory;
	};
	// ModuleName -> SettingsItemUpdateCallback
	std::unordered_map<std::string, RegisteredModuleSettings> RegisteredModules;
	std::mutex RegisteredModulesMutex;

	nosResult RegisterEditorSettings(u64 itemCount, const nosSettingsEditorItem** itemList, nosPfnSettingsItemUpdate itemUpdateCallback, nosSettingsFileDirectory saveDirectory);
	nosResult UnregisterEditorSettings();
	void OnEditorConnected(uint64_t editorId);
	void OnMessageFromEditor(uint64_t editorId, nosBuffer message);
};
}