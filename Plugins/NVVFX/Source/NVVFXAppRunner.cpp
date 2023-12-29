#include "NVVFXAppRunner.h"

NVVFXAppRunner::NVVFXAppRunner() : EffectHandle(NULL)
{
}

NVVFXAppRunner::~NVVFXAppRunner()
{
}

nosResult NVVFXAppRunner::CreateEffect(std::string effectSelector, std::string modelsDir)
{
    if (std::filesystem::exists(modelsDir)) {
        nosEngine.LogE("Models directory %s is not exists", modelsDir.c_str());
        return NOS_RESULT_FAILED;
    }

    NvCV_Status res = NvVFX_CreateEffect(effectSelector.c_str(), &EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetString(EffectHandle, NVVFX_MODEL_DIRECTORY, modelsDir.c_str());
    CHECK_NVCV_ERROR(res);    

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::Run(NvCVImage* input, NvCVImage* output)
{
    if (EffectHandle == NULL)
        return NOS_RESULT_FAILED;

    CUstream stream = 0;

    NvCV_Status res = NvVFX_SetImage(EffectHandle, NVVFX_INPUT_IMAGE, input);
    CHECK_NVCV_ERROR(res);
    
    res = NvVFX_SetImage(EffectHandle, NVVFX_OUTPUT_IMAGE, output);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetCudaStream(EffectHandle, NVVFX_CUDA_STREAM, stream);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetU32(EffectHandle, NVVFX_MODE, ArtifactReduction);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetF32(EffectHandle, NVVFX_STRENGTH, Strength);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_Load(EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_Run(EffectHandle, 0);
    CHECK_NVCV_ERROR(res);


    return NOS_RESULT_SUCCESS;
}

void NVVFXAppRunner::SetArtifactReduction(bool isActive)
{
    ArtifactReduction = (int)isActive;
}

void NVVFXAppRunner::SetStrength(float strength)
{
    Strength = strength;
}
