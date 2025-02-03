/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#ifndef NOS_SETTINGS_SUBSYSTEM_H_INCLUDED
#define NOS_SETTINGS_SUBSYSTEM_H_INCLUDED
#include "Nodos/Types.h"

typedef enum nosSettingsFileDirectory {
	NOS_SETTINGS_FILE_DIRECTORY_LOCAL, // Module's root folder
	NOS_SETTINGS_FILE_DIRECTORY_WORKSPACE, // Engine's config folder
	NOS_SETTINGS_FILE_DIRECTORY_GLOBAL	// AppData folder
} nosSettingsFileDirectory;

typedef nosResult(*nosPfnSettingsItemUpdate)(const char* entryName, nosBuffer itemValue);

#if NOS_HAS_CPLUSPLUS_20
#include "EditorEvents_generated.h"
typedef nos::sys::settings::editor::SettingsEditorItem nosSettingsEditorItem;
#else
typedef void* nosSettingsEditorItem;
#endif

typedef struct nosSettingsEntryParams {
	nosName TypeName;
	nosBuffer Buffer; // Don't forget to free this buffer
	const char* EntryName; // If NULL, "default" will be used
	nosSettingsFileDirectory Directory;
} nosSettingsEntryParams;

typedef struct nosSettingsSubsystem
{
	// parameters->directory will be set to found directory
	// if parameters->typeName == 0, it will be set to first typename found in the closest file
	// parameters->directory is ignored because closest one is chosen
	nosResult(NOSAPI_CALL *ReadSettingsEntry)(nosSettingsEntryParams* parameters);
	// parameters->typeName should be a valid typename
	// if parameters->directory is not set, it will be set to NOS_SETTINGS_FILE_DIRECTORY_LOCAL
	nosResult(NOSAPI_CALL *WriteSettingsEntry)(const nosSettingsEntryParams* parameters);


	// Editor related functions

	nosResult(NOSAPI_CALL *RegisterEditorSettings)(u64 itemCount, const nosSettingsEditorItem** itemList, nosPfnSettingsItemUpdate itemUpdateCallback, nosSettingsFileDirectory saveDirectory);
	nosResult(NOSAPI_CALL *UnregisterEditorSettings)();
} nosSettingsSubsystem;

#pragma region Helper Declarations & Macros
// Make sure these are same with nossys file.
#define NOS_SETTINGS_SUBSYSTEM_NAME "nos.sys.settings"

#define NOS_SETTINGS_SUBSYSTEM_VERSION_MAJOR 0
#define NOS_SETTINGS_SUBSYSTEM_VERSION_MINOR 2

extern struct nosModuleInfo nosSettingsModuleInfo;
extern nosSettingsSubsystem* nosSettings;

#define NOS_SETTINGS_INIT()                                                                                              \
	nosModuleInfo nosSettingsModuleInfo;                                                                                 \
	nosSettingsSubsystem* nosSettings = nullptr;

#define NOS_SETTINGS_IMPORT() NOS_IMPORT_DEP(NOS_SETTINGS_SUBSYSTEM_NAME, nosSettingsModuleInfo, nosSettings)

#pragma endregion


#endif // NOS_SETTINGS_SUBSYSTEM_H_INCLUDED