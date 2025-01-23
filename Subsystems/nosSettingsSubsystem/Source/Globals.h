// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>

namespace nos::sys::settings {
	struct SettingsFileManager;
	struct SettingsEditorManager;
	extern std::unique_ptr<SettingsFileManager> GSettingsFileManager;
	extern std::unique_ptr<SettingsEditorManager> GSettingsEditorManager;
	extern std::unordered_map<uint32_t, std::unique_ptr<nosSettingsSubsystem>> GExportedSubsystemVersions;
}