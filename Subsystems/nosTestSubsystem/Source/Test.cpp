// Copyright MediaZ AS. All Rights Reserved.
#include <mzTestSubsystem/TestSubsystem.h>
#include <MediaZ/SubsystemAPI.h>

MZ_INIT();

extern "C"
{

MZAPI_ATTR mzResult MZAPI_CALL mzExportSubsystem(void** subsystemContext)
{
	*subsystemContext = new TestSubsystem;
	return MZ_RESULT_SUCCESS;
}

MZAPI_ATTR bool MZAPI_CALL mzUnloadSubsystem(void* subsystemContext)
{
	delete static_cast<TestSubsystem*>(subsystemContext);
	return true;
}

}
