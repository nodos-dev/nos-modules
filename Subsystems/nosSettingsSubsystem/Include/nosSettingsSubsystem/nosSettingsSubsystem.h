/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#ifndef NOS_SETTINGS_SUBSYSTEM_H_INCLUDED
#define NOS_SETTINGS_SUBSYSTEM_H_INCLUDED
#include "Nodos/Types.h"

typedef enum nosSettingsFileDirectory {
	NOS_SETTINGS_FILE_DIRECTORY_LOCAL, // Module's root folder
	NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE, // Engine's binary folder
	NOS_SETTINGS_FILE_DIRECTORY_GLOBAL	// AppData folder
} nosSettingsFileDirectory;

typedef void(*nosPfnSettingsItemUpdate)(nosName itemName, nosBuffer itemValue);


struct nosSettingsSubsystem
{
	nosResult(__stdcall* ReadSettings)(nosName typeName, nosBuffer* buffer);
	nosResult(__stdcall* WriteSettings)(nosName typeName, nosBuffer buffer, nosSettingsFileDirectory directory);
	// buffer: It should be nos.sys.settings.editor.SettingsList fbs buffer.
	nosResult(__stdcall* RegisterEditorSettings)(nosBuffer buffer, nosPfnSettingsItemUpdate itemUpdateCallback);
	nosResult(__stdcall* UnregisterEditorSettings)();
};

#pragma region Helper Declarations & Macros
// Make sure these are same with nossys file.
#define NOS_SETTINGS_SUBSYSTEM_NAME "nos.sys.settings"

#define NOS_SETTINGS_SUBSYSTEM_VERSION_MAJOR 0
#define NOS_SETTINGS_SUBSYSTEM_VERSION_MINOR 1

extern struct nosModuleInfo nosSettingsModuleInfo;
extern nosSettingsSubsystem* nosSettings;

#define NOS_SETTINGS_INIT()                                                                                              \
	nosModuleInfo nosSettingsModuleInfo;                                                                                 \
	nosSettingsSubsystem* nosSettings = nullptr;

#define NOS_SETTINGS_IMPORT() NOS_IMPORT_DEP(NOS_SETTINGS_SUBSYSTEM_NAME, nosSettingsModuleInfo, nosSettings)

#pragma endregion


#endif // NOS_SETTINGS_SUBSYSTEM_H_INCLUDED