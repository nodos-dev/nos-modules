// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/Types_generated.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>
#include "SettingsEditorManager.h"
#include "SettingsFileManager.h"
#include "Globals.h"
NOS_INIT()
NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

namespace settings = nos::sys::settings;
namespace nos::sys::settings {
	std::unique_ptr<SettingsFileManager> GSettingsFileManager = nullptr;
	std::unique_ptr<SettingsEditorManager> GSettingsEditorManager = nullptr;
	std::unordered_map<uint32_t, std::unique_ptr<nosSettingsSubsystem>> GExportedSubsystemVersions;
}

nosResult NOSAPI_CALL OnPreUnloadSubsystem()
{
	settings::GSettingsFileManager = nullptr;
	settings::GSettingsEditorManager = nullptr;
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
	nosResult OnRequest(uint32_t minor, void** outSubsystemCtx)
	{
		if (auto it = settings::GExportedSubsystemVersions.find(minor); it != settings::GExportedSubsystemVersions.end())
		{
			*outSubsystemCtx = it->second.get();
			return NOS_RESULT_SUCCESS;
		}
		nosSettingsSubsystem& subsystem = *(settings::GExportedSubsystemVersions[minor] = std::make_unique<nosSettingsSubsystem>());
		subsystem.ReadSettingsEntry = [](nosSettingsEntryParams* params) -> nosResult { return settings::GSettingsFileManager->ReadSettings(params); };
		subsystem.WriteSettingsEntry = [](const nosSettingsEntryParams* params) -> nosResult { return settings::GSettingsFileManager->WriteSettings(params); };
		subsystem.RegisterEditorSettings = [](u64 itemCount, const nosSettingsEditorItem** list, nosPfnSettingsItemUpdate itemUpdateCallback, nosSettingsFileDirectory saveDirectory) { return settings::GSettingsEditorManager->RegisterEditorSettings(itemCount, list, itemUpdateCallback, saveDirectory); };
		subsystem.UnregisterEditorSettings = []() -> nosResult { return settings::GSettingsEditorManager->UnregisterEditorSettings(); };
		*outSubsystemCtx = &subsystem;
		return NOS_RESULT_SUCCESS;
	}

	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
	{
		subsystemFunctions->OnRequest = OnRequest;
		subsystemFunctions->OnPreUnloadSubsystem = OnPreUnloadSubsystem;
		subsystemFunctions->OnEditorConnected = [](uint64_t editorId) {settings::GSettingsEditorManager->OnEditorConnected(editorId); };
		subsystemFunctions->OnMessageFromEditor = [](uint64_t editorId, nosBuffer message) { settings::GSettingsEditorManager->OnMessageFromEditor(editorId, message); };
		settings::GSettingsFileManager.reset(new settings::SettingsFileManager());
		settings::GSettingsEditorManager.reset(new settings::SettingsEditorManager());

		return NOS_RESULT_SUCCESS;
	}
}