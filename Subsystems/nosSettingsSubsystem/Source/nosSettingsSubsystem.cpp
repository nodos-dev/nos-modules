// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosSettingsSubsystem/nosSettingsSubsystem.h>
#include <nosSettingsSubsystem/Types_generated.h>
#include <nosSettingsSubsystem/EditorEvents_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>
NOS_INIT()
NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

std::filesystem::path GetAppDataFolder();
NOS_REGISTER_NAME_SPACED(SETTINGS_ENTRY_TYPENAME, nos::sys::settings::SettingsEntry::GetFullyQualifiedName());

struct SettingsDataManager;
static std::unique_ptr<SettingsDataManager> GSettingsDataManager = nullptr;
namespace settingsFb = nos::sys::settings;
static std::unordered_map<uint32_t, std::unique_ptr<nosSettingsSubsystem>> GExportedSubsystemVersions;

static constexpr nosSettingsFileDirectory DirectoriesClosestToFarthest[] = {
	NOS_SETTINGS_FILE_DIRECTORY_LOCAL,
	NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE,
	NOS_SETTINGS_FILE_DIRECTORY_GLOBAL
};

std::string GetSettingsFileName(nosModuleInfo const& module) {
	return std::string(nosEngine.GetString(module.Id.Name)) + "-" + std::string(nosEngine.GetString(module.Id.Version));
}

struct SettingsDataManager {
	struct SettingsFile {
		std::string TypeName;
		nos::Buffer Data;
		nosSettingsFileDirectory Directory;
		nosResult Save(std::filesystem::path filePath, const nosModuleInfo& info) const {
			std::ofstream file(filePath);
			flatbuffers::FlatBufferBuilder builder;

			if (!file || !file.is_open()) {
				nosEngine.LogE("Settings subsystem couldn't open file for writing");
				return NOS_RESULT_FAILED;
			}

			flatbuffers::FlatBufferBuilder fbb;
			::flatbuffers::Offset<nos::sys::settings::SettingsEntry> settings;
			{
				std::vector<uint8_t> data = Data;
				settings = settingsFb::CreateSettingsEntryDirect
				(
					fbb, nos::Name(info.Id.Name).AsCStr(),
					nos::Name(info.Id.Version).AsCStr(),
					TypeName.c_str(),
					&data
				);
			}

			fbb.Finish(settings);
			nos::Buffer data = fbb.Release();

			char* json = nullptr;
			auto ret = nosEngine.GenerateJsonFromBuffer(NSN_SETTINGS_ENTRY_TYPENAME, data.GetInternal(), &json);
			if (ret != NOS_RESULT_SUCCESS) {
				nosEngine.LogE("Failed to generate JSON from buffer for nosSettingsSubsystem");
				return ret;
			}

			file << json;
			file.close();
			return NOS_RESULT_SUCCESS;
		}
	};
	// Each module has a settings file
	// ModuleName-Version -> SettingsFile
	std::unordered_map<std::string, std::vector<SettingsFile>> SettingsFiles;
	std::mutex SettingsFilesMutex;
	SettingsFile& FindOrCreateSettingsFile(nosModuleInfo const& module, nosName typeName, nosSettingsFileDirectory dir) {
		std::string fileName = GetSettingsFileName(module);
		auto& fileList = GSettingsDataManager->SettingsFiles[fileName];
		auto it = std::find_if(fileList.begin(), fileList.end(), [dir](auto& file) { return file.Directory == dir; });
		if (it == fileList.end()) {
			fileList.push_back({ nos::Name(typeName).AsString(), nullptr, dir });
			return fileList.back();
		}
		else {
			return *it;
		}
	}

	struct RegisteredModuleSettings {
		nos::Buffer settingsList;
		nosPfnSettingsItemUpdate itemUpdateCallback = nullptr;
	};
	// ModuleName -> SettingsItemUpdateCallback
	std::unordered_map<std::string, RegisteredModuleSettings> RegisteredModules;
	std::mutex RegisteredModulesMutex;

	static nosResult ReadSettingsFile(std::filesystem::path filePath, SettingsFile& file) {
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
		if (nosEngine.GenerateBufferFromJson(NSN_SETTINGS_ENTRY_TYPENAME, json, &data) != NOS_RESULT_SUCCESS) {
			nosEngine.LogE("Failed to generate buffer from JSON for nosSettingsSubsystem");
			return NOS_RESULT_FAILED;
		}


		const auto& rootTable = flatbuffers::GetRoot<nos::sys::settings::SettingsEntry>(data.Data);
		flatbuffers::Verifier verifier((uint8_t*)data.Data, data.Size);
		if (!rootTable->Verify(verifier)) {
			nosEngine.LogW("Failed to verify the settings file: %s", filePath.c_str());
			nosEngine.FreeBuffer(&data);
			free(json);
			return NOS_RESULT_FAILED;
		}

		file.Data = rootTable->data();
		nosEngine.FreeBuffer(&data);
		free(json);
		return NOS_RESULT_SUCCESS;
	}

	static std::filesystem::path GetSettingsFilePath(nosSettingsFileDirectory dir, const nosModuleInfo& moduleInfo) {
		std::string fileName = GetSettingsFileName(moduleInfo) + ".json";
		switch (dir)
		{
		case NOS_SETTINGS_FILE_DIRECTORY_LOCAL:
		{
			return std::filesystem::path(moduleInfo.RootFolderPath) / fileName;
		}
		case NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE:
		{
			return std::filesystem::path(nosEngine.Module->RootFolderPath) / fileName;
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
	
	static nosResult ReadSettings(nosName typeName, nosBuffer* settingsBuffer) {
		nosModuleInfo module = {};
		if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;
		auto fileName = GetSettingsFileName(module);
		std::unique_lock lock(GSettingsDataManager->SettingsFilesMutex);
		// From closest to farthest, try to read the settings file
		for (uint32_t dirIndx = 0; dirIndx < sizeof(DirectoriesClosestToFarthest) / sizeof(DirectoriesClosestToFarthest[0]); dirIndx++) {
			auto dir = DirectoriesClosestToFarthest[dirIndx];
			SettingsFile& settings = GSettingsDataManager->FindOrCreateSettingsFile(module, typeName, dir);

			auto readFileResult = ReadSettingsFile(GetSettingsFilePath(dir, module), settings);
			if (readFileResult != NOS_RESULT_SUCCESS)
				continue;

			*settingsBuffer = *settings.Data.GetInternal();
			return NOS_RESULT_SUCCESS;
		}
		nosEngine.LogE("Settings not found for module %s", nosEngine.GetString(module.Id.Name));
		return NOS_RESULT_FAILED;
	}

	static nosResult WriteSettings(nosName typeName, nosBuffer settingsBuffer, nosSettingsFileDirectory directory) {
		nosModuleInfo module = {};
		if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;
		std::unique_lock lock(GSettingsDataManager->SettingsFilesMutex);
		SettingsFile& settings = GSettingsDataManager->FindOrCreateSettingsFile(module, typeName, directory);
		settings.Data = settingsBuffer;
		settings.TypeName = nos::Name(typeName);
		return settings.Save(GetSettingsFilePath(directory, module), module);
	}

	static nosResult RegisterEditorSettings(nosBuffer buffer, nosPfnSettingsItemUpdate itemUpdateCallback) {
		nosModuleInfo module = {};
		if (nosEngine.GetCallingModule(&module) != NOS_RESULT_SUCCESS)
			return NOS_RESULT_FAILED;

		auto rootTable = flatbuffers::GetRoot<settingsFb::editor::SettingsList>(buffer.Data);
		flatbuffers::Verifier verifier((uint8_t*)buffer.Data, buffer.Size);
		if (!rootTable->Verify(verifier)) {
			nosEngine.LogW("Module %s sent invalid buffer of %s", nos::Name(module.Id.Name).AsCStr(), settingsFb::editor::SettingsList::GetFullyQualifiedName());
			return NOS_RESULT_FAILED;
		}
		if (itemUpdateCallback == nullptr) {
			nosEngine.LogE("Module %s sent a null callback for RegisterEditorSettings", nos::Name(module.Id.Name).AsCStr());
			return NOS_RESULT_FAILED;
		}

		std::unique_lock lock(GSettingsDataManager->RegisteredModulesMutex);
		auto& registeredModuleInfo = GSettingsDataManager->RegisteredModules[nos::Name(module.Id.Name).AsString()];
		registeredModuleInfo.itemUpdateCallback = itemUpdateCallback;
		registeredModuleInfo.settingsList = buffer;
		nosEngine.SendCustomMessageToEditors(NOS_NAME_STATIC(NOS_SETTINGS_SUBSYSTEM_NAME), buffer);

		return NOS_RESULT_SUCCESS;
	}
	static nosResult UnregisterEditorSettings() {
		std::unique_lock lock(GSettingsDataManager->RegisteredModulesMutex);
		if (auto it = GSettingsDataManager->RegisteredModules.find(nos::Name(nosEngine.Module->Id.Name).AsString()); it != GSettingsDataManager->RegisteredModules.end())
		{
			GSettingsDataManager->RegisteredModules.erase(it);
			return NOS_RESULT_SUCCESS;
		}
		nosEngine.LogE("Module %s is not registered for editor settings, but called UnregisterEditorSettings().", nosEngine.GetString(nosEngine.Module->Id.Name));
		return NOS_RESULT_FAILED;
	}
	static void OnEditorConnected(uint64_t editorId) {
		for (auto& [moduleName, registeredModuleInfo] : GSettingsDataManager->RegisteredModules) {
			nosEngine.SendCustomMessageToEditors(NOS_NAME_STATIC(NOS_SETTINGS_SUBSYSTEM_NAME), registeredModuleInfo.settingsList);
		}
	}
	static void OnMessageFromEditor(uint64_t editorId, nosBuffer message) {
		auto& regModules = GSettingsDataManager->RegisteredModules;
		auto msg = flatbuffers::GetRoot<settingsFb::editor::SettingsUpdateFromEditor>(message.Data);
		auto moduleName = msg->module_name()->c_str();
		auto it = regModules.find(moduleName);
		if (it == regModules.end()) {
			nosEngine.LogE("Module %s is not registered for editor settings, but Editor sent a message.", moduleName);
			return;
		}

		it->second.itemUpdateCallback(nos::Name(msg->item_name()->c_str()).ID, *nos::Buffer(msg->data()).GetInternal());
	}

	static nosResult OnRequest(uint32_t minor, void** outSubsystemCtx)
	{
		if (auto it = GExportedSubsystemVersions.find(minor); it != GExportedSubsystemVersions.end())
		{
			*outSubsystemCtx = it->second.get();
			return NOS_RESULT_SUCCESS;
		}
		nosSettingsSubsystem& subsystem = *(GExportedSubsystemVersions[minor] = std::make_unique<nosSettingsSubsystem>());
		subsystem.ReadSettings = ReadSettings;
		subsystem.WriteSettings = WriteSettings;
		subsystem.RegisterEditorSettings = RegisterEditorSettings;
		subsystem.UnregisterEditorSettings = UnregisterEditorSettings;
		*outSubsystemCtx = &subsystem;
		return NOS_RESULT_SUCCESS;
	}
};


nosResult NOSAPI_CALL OnPreUnloadSubsystem()
{
	GSettingsDataManager = nullptr;
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
	{
		subsystemFunctions->OnRequest = SettingsDataManager::OnRequest;
		subsystemFunctions->OnPreUnloadSubsystem = OnPreUnloadSubsystem;
		subsystemFunctions->OnEditorConnected = SettingsDataManager::OnEditorConnected;
		subsystemFunctions->OnMessageFromEditor = SettingsDataManager::OnMessageFromEditor;
		GSettingsDataManager.reset(new SettingsDataManager());

		return NOS_RESULT_SUCCESS;
	}
}



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