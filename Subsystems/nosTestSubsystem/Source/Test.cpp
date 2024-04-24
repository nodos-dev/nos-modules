// Copyright Nodos AS. All Rights Reserved.
#include <nosTestSubsystem/TestSubsystem.h>
#include <Nodos/SubsystemAPI.h>

NOS_INIT();

extern "C"
{


NOSAPI_ATTR nosResult NOSAPI_CALL OnRequest(uint32_t minor, void** outSubsystemContext)
{
	static TestSubsystem testSubsystem = {};
	*outSubsystemContext = &testSubsystem;
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
{
	subsystemFunctions->OnRequest = OnRequest;
	return NOS_RESULT_SUCCESS;
}

}
