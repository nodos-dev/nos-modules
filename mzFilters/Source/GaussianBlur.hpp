#pragma once

#include <MediaZ/PluginAPI.h>

bool MZAPI_CALL GaussianBlur_ExecuteNode(void* ctx, const MzNodeExecuteArgs* args);
void MZAPI_CALL GaussianBlur_OnNodeCreated(const MzFbNode*, void** outCtxPtr);