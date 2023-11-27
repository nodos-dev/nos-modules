// Copyright Nodos AS. All Rights Reserved.
#include <nosTestSubsystem/TestSubsystem.h>
#include <Nodos/SubsystemAPI.h>

NOS_INIT();

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(void** subsystemContext)
{
	*subsystemContext = new TestSubsystem;
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR bool NOSAPI_CALL nosUnloadSubsystem(void* subsystemContext)
{
	delete static_cast<TestSubsystem*>(subsystemContext);
	return true;
}

}
