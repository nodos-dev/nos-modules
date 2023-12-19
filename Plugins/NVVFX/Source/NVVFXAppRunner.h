#pragma once
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>
#include <iostream>

#include "nvVideoEffects.h"
#include <Windows.h>
#include "Nodos/PluginAPI.h"
#include "Nodos/PluginHelpers.hpp"
#include "CUDAResourceManager.h"

class NVVFXAppRunner {
public:
	NVVFXAppRunner(std::string modelsDir);
	~NVVFXAppRunner();

	nosResult CreateEffect(const char* effect);
	void SetInputImage(NvCVImage& image);

	NvCVImage InputGPUBuf;
	NvCVImage OutputGPUBuf;
private:
	std::string ModelsDirectory;
	CudaGPUResourceManager ResourceManager;
};