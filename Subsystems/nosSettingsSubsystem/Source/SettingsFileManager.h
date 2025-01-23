// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/Types_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>

namespace nos::sys::settings {
	struct SettingsFileManager {
		struct SettingsFile {
			// EntryName -> {TypeName, Buffer}
			std::unordered_map<std::string, std::pair<std::string, nos::Buffer>> Entries;
			nosSettingsFileDirectory Directory;
		};
		// Each module has a settings file
		// ModuleName-Version -> [SettingsFile]
		std::unordered_map<std::string, std::vector<SettingsFile>> SettingsFiles;
		std::mutex SettingsFilesMutex;
		SettingsFile& FindOrCreateSettingsFile(nosModuleInfo const& module, nosSettingsFileDirectory dir);
		std::filesystem::path GetSettingsFilePath(nosSettingsFileDirectory dir, const nosModuleInfo& moduleInfo) const;
		nosResult ReadSettingsFile(std::filesystem::path filePath, SettingsFile& file);
		nosResult ReadSettings(nosSettingsEntryParams* params);
		nosResult WriteSettings(const nosSettingsEntryParams* params);
		nosResult WriteSettings(const nosSettingsEntryParams* params, const nosModuleInfo& callingModule);
		nosResult WriteSettingsFile(const SettingsFile& settingsFile, const nosModuleInfo& info) const;
	};
}