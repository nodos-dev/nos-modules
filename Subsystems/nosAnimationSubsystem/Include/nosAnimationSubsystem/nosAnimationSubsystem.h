/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#ifndef NOS_ANIMATION_SUBSYSTEM_H_INCLUDED
#define NOS_ANIMATION_SUBSYSTEM_H_INCLUDED
#include "Nodos/Types.h"

typedef nosResult(*nosPfnAnimateData)(const nosBuffer from, const nosBuffer to, double t, nosBuffer* outBuf);

struct nosAnimationHandler
{
	nosName TypeName;
	nosPfnAnimateData AnimateData;
};

typedef struct nosAnimationSubsystem
{
	nosResult(NOSAPI_CALL* RegisterAnimationHandler)(nosAnimationHandler const* handler);
	nosResult(NOSAPI_CALL* HasAnimationHandler)(nosName typeName, bool* hasHandler);
	nosResult(NOSAPI_CALL* AnimateData)(nosName typeName, const nosBuffer from, const nosBuffer to, double t, nosBuffer* outBuf);
} nosAnimationSubsystem;

#pragma region Helper Declarations & Macros
// Make sure these are same with nossys file.
#define NOS_ANIMATION_SUBSYSTEM_NAME "nos.sys.animation"

#define NOS_ANIMATION_SUBSYSTEM_VERSION_MAJOR 1
#define NOS_ANIMATION_SUBSYSTEM_VERSION_MINOR 3

extern struct nosModuleInfo nosAnimationModuleInfo;
extern nosAnimationSubsystem* nosAnimation;

#define NOS_ANIMATION_INIT()                                                                                              \
	nosModuleInfo nosAnimationModuleInfo;                                                                                 \
	nosAnimationSubsystem* nosAnimation = nullptr;

#define NOS_ANIMATION_IMPORT() NOS_IMPORT_DEP(NOS_ANIMATION_SUBSYSTEM_NAME, nosAnimationModuleInfo, nosAnimation)

#pragma endregion


#endif // NOS_ANIMATION_SUBSYSTEM_H_INCLUDED