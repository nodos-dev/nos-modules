/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#ifndef NOS_CONFIG_SUBSYSTEM_H_INCLUDED
#define NOS_CONFIG_SUBSYSTEM_H_INCLUDED
#include "Nodos/Types.h"

typedef enum nosConfigurationFileLocation {
	NOS_CONFIG_FILE_LOCATION_LOCAL,
	NOS_CONFIG_FILE_LOCATION_APPDATA
} nosConfigurationFileLocation;

struct nosConfigSubsystem
{
	nosResult(__stdcall* ReadConfig)(nosModuleInfo const* module, nosName typeName, const char** configText);
	nosResult(__stdcall* WriteConfig)(nosModuleInfo const* module, nosName typeName, const char* configText);
	nosResult(__stdcall* Save)(nosConfigurationFileLocation location);
};

#endif // NOS_CONFIG_SUBSYSTEM_H_INCLUDED