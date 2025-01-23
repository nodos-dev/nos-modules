// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/Types_generated.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>
#include "SettingsFileManager.h"
#include "Globals.h"

namespace nos::sys::settings {
std::filesystem::path GetAppDataFolder();
NOS_REGISTER_NAME_SPACED(MODULE_SETTINGS_TYPENAME, nos::sys::settings::ModuleSettings::GetFullyQualifiedName());
static constexpr char DEFAULT_SETTINGS_ENTRY_NAME[] = "default";

static constexpr nosSettingsFileDirectory DIRECTORIES_CLOSEST_TO_FARTHEST[] = {
	NOS_SETTINGS_FILE_DIRECTORY_LOCAL,
	NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE,
	NOS_SETTINGS_FILE_DIRECTORY_GLOBAL
};

std::string GetSettingsFileName(nosModuleInfo const& module)
{
	return std::string(nosEngine.GetString(module.Id.Name)) + "-" + std::string(nosEngine.GetString(module.Id.Version));
}

std::filesystem::path SettingsFileManager::GetSettingsFilePath(nosSettingsFileDirectory dir, const nosModuleInfo& moduleInfo) const
{
	std::string fileName = GetSettingsFileName(moduleInfo) + ".json";
	switch (dir)
	{
	case NOS_SETTINGS_FILE_DIRECTORY_LOCAL:
	{
		return std::filesystem::path(moduleInfo.RootFolderPath) / fileName;
	}
	case NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE:
	{
		return fileName;
	}
	case NOS_SETTINGS_FILE_DIRECTORY_GLOBAL:
	{
		return GetAppDataFolder() / fileName;
	}
	default:
		nosEngine.LogE("The file location requested is not supported by nosSettingsSubsystem.");
		return std::filesystem::path();
	}
}

SettingsFileManager::SettingsFile& SettingsFileManager::FindOrCreateSettingsFile(nosModuleInfo const& module, nosSettingsFileDirectory dir)
{
	std::string fileName = GetSettingsFileName(module);
	auto& fileList = SettingsFiles[fileName];
	auto it = std::find_if(fileList.begin(), fileList.end(), [dir](auto& file) { return file.Directory == dir; });
	if (it == fileList.end()) {
		fileList.insert(fileList.begin() + dir, { {}, dir });
		return fileList.back();
	}
	else {
		return *it;
	}
}

nosResult SettingsFileManager::ReadSettingsFile(std::filesystem::path filePath, SettingsFile& file)
{
	char* json = nullptr;
	// Read to JSON
	{
		std::ifstream file(filePath);
		if (!file || !file.is_open()) {
			return NOS_RESULT_FAILED;
		}

		std::stringstream buffer;
		buffer << file.rdbuf();  // Read the entire file into the buffer
		file.close();

		std::string content = buffer.str();  // Convert buffer to string
		json = (char*)malloc(content.size() + 1); // Allocate memory (+1 for null terminator)
		std::memcpy(json, content.c_str(), content.size() + 1); // Copy content to json
	}

	nosBuffer data = {};
	if (nosEngine.GenerateBufferFromJson(NSN_MODULE_SETTINGS_TYPENAME, json, &data) != NOS_RESULT_SUCCESS) {
		nosEngine.LogE("Failed to generate buffer from JSON for nosSettingsSubsystem");
		return NOS_RESULT_FAILED;
	}


	const auto& rootTable = flatbuffers::GetRoot<nos::sys::settings::ModuleSettings>(data.Data);
	flatbuffers::Verifier verifier((uint8_t*)data.Data, data.Size);
	if (!rootTable->Verify(verifier)) {
		nosEngine.LogW("Failed to verify the settings file: %s", filePath.c_str());
		nosEngine.FreeBuffer(&data);
		free(json);
		return NOS_RESULT_FAILED;
	}

	file.Entries.clear();
	if (rootTable->settings()) {
		for (const auto& entry : *rootTable->settings()) {
			file.Entries[entry->entry_name()->str()] = { entry->type_name()->str(), entry->data() };
		}
	}

	nosEngine.FreeBuffer(&data);
	free(json);
	return NOS_RESULT_SUCCESS;
}

nosResult SettingsFileManager::ReadSettings(nosSettingsEntryParams* params)
{
	nosModuleInfo module = {};
	if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
		return NOS_RESULT_FAILED;
	auto fileName = GetSettingsFileName(module);
	std::unique_lock lock(SettingsFilesMutex);
	// From closest to farthest, try to read the settings file
	for (uint32_t dirIndx = 0; dirIndx < sizeof(DIRECTORIES_CLOSEST_TO_FARTHEST) / sizeof(DIRECTORIES_CLOSEST_TO_FARTHEST[0]); dirIndx++) {
		auto dir = DIRECTORIES_CLOSEST_TO_FARTHEST[dirIndx];
		SettingsFile& settings = FindOrCreateSettingsFile(module, dir);

		auto readFileResult = ReadSettingsFile(GetSettingsFilePath(dir, module), settings);
		if (readFileResult != NOS_RESULT_SUCCESS)
			continue;

		// TypeName, Data
		std::pair<std::string, nos::Buffer>* entryData = nullptr;
		if (params->EntryName == nullptr)
		{
			if (!settings.Entries.size())
				continue;
			entryData = &settings.Entries.begin()->second;
		}
		else {
			auto entryIt = settings.Entries.find(params->EntryName);
			if (entryIt == settings.Entries.end())
				continue;
			entryData = &entryIt->second;
		}

		auto typeName = nos::Name(entryData->first);
		if (params->TypeName.ID && params->TypeName != typeName) {
			nosEngine.LogE("Settings type mismatch for module %s for entry %s", nosEngine.GetString(module.Id.Name), params->EntryName);
			return NOS_RESULT_FAILED;
		}
		auto& localBuf = entryData->second;
		auto ret = nosEngine.AllocateBuffer(localBuf.Size(), &params->Buffer);
		if (ret != NOS_RESULT_SUCCESS) {
			nosEngine.LogE("Failed to allocate buffer for settings entry %s", params->EntryName);
			return ret;
		}
		params->TypeName = typeName;

		memcpy(params->Buffer.Data, localBuf.Data(), localBuf.Size());
		return NOS_RESULT_SUCCESS;
	}
	nosEngine.LogE("Settings not found for module %s", nosEngine.GetString(module.Id.Name));
	return NOS_RESULT_FAILED;
}

nosResult SettingsFileManager::WriteSettings(const nosSettingsEntryParams* params)
{
	nosModuleInfo module = {};
	if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
		return NOS_RESULT_FAILED;
	return WriteSettings(params, module);
}

nosResult SettingsFileManager::WriteSettings(const nosSettingsEntryParams* params, const nosModuleInfo& module)
{
	std::unique_lock lock(SettingsFilesMutex);
	SettingsFile& settings = FindOrCreateSettingsFile(module, params->Directory);
	auto& entry = settings.Entries[params->EntryName ? params->EntryName : DEFAULT_SETTINGS_ENTRY_NAME];
	entry.first = nos::Name(params->TypeName);
	entry.second = params->Buffer;
	return WriteSettingsFile(settings, module);
}

nosResult SettingsFileManager::WriteSettingsFile(const SettingsFile& settingsFile, const nosModuleInfo& info) const
{
	std::ofstream file(GetSettingsFilePath(settingsFile.Directory, info));
	flatbuffers::FlatBufferBuilder builder;

	if (!file || !file.is_open()) {
		nosEngine.LogE("Settings subsystem couldn't open file for writing");
		return NOS_RESULT_FAILED;
	}


	nos::sys::settings::TModuleSettings moduleSettings;
	moduleSettings.module_name = nos::Name(info.Id.Name).AsString();
	moduleSettings.module_version = nos::Name(info.Id.Version).AsString();
	for (auto& it : settingsFile.Entries) {
		moduleSettings.settings.push_back(std::unique_ptr<nos::sys::settings::TSettingsEntry>(new nos::sys::settings::TSettingsEntry()));
		auto& tEntry = moduleSettings.settings.back();
		tEntry->data = it.second.second;
		tEntry->entry_name = it.first;
		tEntry->type_name = it.second.first;
	}

	auto data = nos::Buffer::From(moduleSettings);

	char* json = nullptr;
	auto ret = nosEngine.GenerateJsonFromBuffer(NSN_MODULE_SETTINGS_TYPENAME, data.GetInternal(), &json);
	if (ret != NOS_RESULT_SUCCESS) {
		nosEngine.LogE("Failed to generate JSON from buffer for nosSettingsSubsystem");
		return ret;
	}

	file << json;
	file.close();
	return NOS_RESULT_SUCCESS;
};

#if defined(_WIN32)
std::filesystem::path GetAppDataFolder()
{
	std::filesystem::path appDataPath;
	char* appData = nullptr;
	size_t len = 0;
	if (_dupenv_s(&appData, &len, "APPDATA") == 0 && appData != nullptr)
	{
		appDataPath = std::filesystem::path(appData);
		free(appData);
	}
	return appDataPath;
}
#elif defined(__linux__)
std::filesystem::path GetAppDataFolder()
{
	std::filesystem::path appDataPath;
	const char* home = std::getenv("HOME");
	if (home)
	{
		appDataPath = std::filesystem::path(home) / ".local" / "share";
	}
	return appDataPath;
}
#endif

}