/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#ifndef NOS_ANIMATION_SUBSYSTEM_H_INCLUDED
#define NOS_ANIMATION_SUBSYSTEM_H_INCLUDED
#include "Nodos/Types.h"

typedef nosResult(*nosAnimationInterpolateCallback)(const nosBuffer from, const nosBuffer to, const double t, nosBuffer* outBuf);

typedef struct nosAnimationInterpolator
{
	nosName TypeName;
	nosAnimationInterpolateCallback InterpolateCallback;
} nosAnimationInterpolator;

typedef struct nosAnimationSubsystem
{
	nosResult(NOSAPI_CALL* RegisterInterpolator)(nosAnimationInterpolator const* interpolator);
	nosResult(NOSAPI_CALL* HasInterpolator)(nosName typeName, bool* hasInterpolator);
	/// If result is NOS_RESULT_SUCCESS, then outBuf is a buffer allocated using nosEngine.AllocateBuffer
	/// which should be freed with nosEngine.FreeBuffer by the calling module.
	nosResult(NOSAPI_CALL* Interpolate)(nosName typeName, const nosBuffer from, const nosBuffer to, const double t, nosBuffer* outBuf);
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