// Copyright Nodos AS. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <nosConfigurationSubsystem/nosConfigurationSubsystem.h>
#include <nosConfigurationSubsystem/Types_generated.h>
#include <unordered_map>
#include <Nodos/Helpers.hpp>
NOS_INIT()
NOS_BEGIN_IMPORT_DEPS()
NOS_END_IMPORT_DEPS()

auto const GConfigFileName = "nosConfigurationSubsystem.json";
std::filesystem::path GetAppDataFolder();

struct ConfigDataManager {
	struct ConfigFile {
		std::unordered_map<std::string, std::string> ConfigData;
	};
	std::unordered_map<nosConfigurationFileLocation, ConfigFile> ConfigFiles;
	static std::filesystem::path GetConfigFileLocation(nosConfigurationFileLocation path) {
		switch (path)
		{
		case NOS_CONFIG_FILE_LOCATION_LOCAL:
		{
			static const auto path = std::filesystem::path(nosEngine.Module->RootFolderPath) / GConfigFileName;
			return path;
		}
		case NOS_CONFIG_FILE_LOCATION_APPDATA:
		{
			static const auto path = GetAppDataFolder() / GConfigFileName;
			return path;
		}
		default:
			nosEngine.LogE("The file location requested is not supported by nosConfigurationSubsystem.");
			return std::filesystem::path();
		}

	}
};
static const nosConfigurationFileLocation ConfigFileLocationsFromNearestToFarthest[] = {
	NOS_CONFIG_FILE_LOCATION_LOCAL,
	NOS_CONFIG_FILE_LOCATION_APPDATA
};

static ConfigDataManager* GConfigDataManager = nullptr;
namespace configFb = nos::sys::config;


std::string GetConfigKey(nosModuleInfo const* module) {
	return std::string(nosEngine.GetString(module->Id.Name)) + "-" + std::string(nosEngine.GetString(module->Id.Version));
}

nosName moduleConfigListTypeName = 0;
nosResult ReadConfigFile(nosConfigurationFileLocation filePath, std::unordered_map<std::string, std::string>& configs) {
	char* json = nullptr;
	// Read to JSON
	{
		std::ifstream file(ConfigDataManager::GetConfigFileLocation(filePath));
		if (!file || !file.is_open()) {
			nosEngine.LogE("Configuration subsystem couldn't open file for reading");
			return NOS_RESULT_FAILED;
		}

		std::stringstream buffer;
		buffer << file.rdbuf();  // Read the entire file into the buffer
		file.close();

		std::string content = buffer.str();  // Convert buffer to string
		json = new char[content.size() + 1]; // Allocate memory (+1 for null terminator)
		std::memcpy(json, content.c_str(), content.size() + 1); // Copy content to json
	}

	nosBuffer data = {};
	if (nosEngine.GenerateBufferFromJson(moduleConfigListTypeName, json, &data) != NOS_RESULT_SUCCESS) {
		nosEngine.LogE("Failed to generate buffer from JSON for nosConfigurationSubsystem");
		return NOS_RESULT_FAILED;
	}


	const auto& rootTable = flatbuffers::GetRoot<nos::sys::config::ModuleConfigList>(data.Data);
	flatbuffers::Verifier verifier((uint8_t*)data.Data, data.Size);
	if (!rootTable->Verify(verifier)) {
		nosEngine.LogE("Failed to verify the config file");
		nosEngine.FreeBuffer(&data);
		free(json);
		return NOS_RESULT_FAILED;
	}

	configs.clear();
	for (auto config : *rootTable->configs()) {
		auto configKey = config->name_and_version()->str();
		auto configData = config->data()->str();
		configs[configKey] = configData;
	}

	nosEngine.FreeBuffer(&data);
	free(json);
	return NOS_RESULT_SUCCESS;
}
nosResult SaveConfigFile(nosConfigurationFileLocation filePath) {
	const auto& configFile = GConfigDataManager->ConfigFiles[filePath];
	std::ofstream file(ConfigDataManager::GetConfigFileLocation(filePath));
	flatbuffers::FlatBufferBuilder builder;

	if (!file || !file.is_open()) {
		nosEngine.LogE("Configuration subsystem couldn't open file for writing");
		return NOS_RESULT_FAILED;
	}

	flatbuffers::FlatBufferBuilder fbb;
	std::vector<::flatbuffers::Offset<nos::sys::config::ConfigEntry>> configs;
	for (auto& [configKey, config] : configFile.ConfigData) {
		configs.push_back(
			configFb::CreateConfigEntry(
				fbb,
				fbb.CreateString(configKey.c_str()),
				fbb.CreateString(config)
			)
		);
	}
	auto configList = configFb::CreateModuleConfigListDirect(fbb, &configs);
	fbb.Finish(configList);
	nos::Buffer data = fbb.Release();

	char* json = nullptr;
	auto ret = nosEngine.GenerateJsonFromBuffer(moduleConfigListTypeName, data.GetInternal(), &json);
	if (ret != NOS_RESULT_SUCCESS) {
		nosEngine.LogE("Failed to generate JSON from buffer for nosConfigurationSubsystem");
		return ret;
	}

	file << json;
	file.close();
	return NOS_RESULT_SUCCESS;
}

nosResult ReadConfig(nosModuleInfo const* module, nosName typeName, const char** configText) {
	for (uint32_t locationIndx = 0; locationIndx < sizeof(ConfigFileLocationsFromNearestToFarthest) / sizeof(ConfigFileLocationsFromNearestToFarthest[0]); locationIndx++) {
		auto location = ConfigFileLocationsFromNearestToFarthest[locationIndx];
		auto& configFile = GConfigDataManager->ConfigFiles[location];
		auto readFileResult = ReadConfigFile(location, configFile.ConfigData);
		if (readFileResult != NOS_RESULT_SUCCESS)
			continue;

		auto it = configFile.ConfigData.find(GetConfigKey(module));
		if (it == configFile.ConfigData.end())
			continue;

		*configText = it->second.c_str();
		return NOS_RESULT_SUCCESS;
	}
	nosEngine.LogE("Configuration not found for module %s", nosEngine.GetString(module->Id.Name));
	return NOS_RESULT_FAILED;
}
nosResult WriteConfig(nosModuleInfo const* module, nosName typeName, const char* configText) {
	auto nearestConfigFile = ConfigFileLocationsFromNearestToFarthest[0];
	GConfigDataManager->ConfigFiles[nearestConfigFile].ConfigData[GetConfigKey(module)] = configText;
	return SaveConfigFile(nearestConfigFile);
}

static std::unordered_map<uint32_t, std::unique_ptr<nosConfigSubsystem>> GExportedSubsystemVersions;

nosResult OnRequest(uint32_t minor, void** outSubsystemCtx)
{
	if (auto it = GExportedSubsystemVersions.find(minor); it != GExportedSubsystemVersions.end())
	{
		*outSubsystemCtx = it->second.get();
		return NOS_RESULT_SUCCESS;
	}
	nosConfigSubsystem& subsystem = *(GExportedSubsystemVersions[minor] = std::make_unique<nosConfigSubsystem>());
	subsystem.ReadConfig = ReadConfig;
	subsystem.WriteConfig = WriteConfig;
	subsystem.Save = SaveConfigFile;
	*outSubsystemCtx = &subsystem;
	return NOS_RESULT_SUCCESS;
}

nosResult NOSAPI_CALL OnPreUnloadSubsystem()
{
	for (auto& configFile : GConfigDataManager->ConfigFiles)
	{
		SaveConfigFile(configFile.first);
	}
	delete GConfigDataManager;
	GConfigDataManager = nullptr;
	return NOS_RESULT_SUCCESS;
}

extern "C"
{
	NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
	{
		subsystemFunctions->OnRequest = OnRequest;
		subsystemFunctions->OnPreUnloadSubsystem = OnPreUnloadSubsystem;
		GConfigDataManager = new ConfigDataManager();

		moduleConfigListTypeName = NOS_NAME_STATIC(nos::sys::config::ModuleConfigList::GetFullyQualifiedName());
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