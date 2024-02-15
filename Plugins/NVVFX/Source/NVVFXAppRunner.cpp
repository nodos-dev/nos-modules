#include "NVVFXAppRunner.h"
#include "cuda.h"

NVVFXAppRunner::NVVFXAppRunner() : AR_EffectHandle(NULL), UpScale_EffectHandle(NULL), SuperRes_EffectHandle(NULL), 
                                   AIGreenScreen_EffectHandle(NULL), AIGS_StateObjectHandle(NULL)
{
    nosCUDA->CreateStream(&stream);
    //nosCUDAContext currentCtx = {};
    //nosCUDA->GetCurrentContext(&currentCtx);
    //cuCtxSetCurrent(reinterpret_cast<CUcontext>(currentCtx));
}

NVVFXAppRunner::~NVVFXAppRunner()
{
    nosCUDA->DestroyStream(stream);
}

nosResult NVVFXAppRunner::InitTransferBuffers(NvCVImage* source, NvCVImage* destination)
{
    NvCV_Status res = NVCV_SUCCESS;
    res = NvCVImage_Init(&InputTransferred, source->width, source->height, source->pitch, source->pixels, source->pixelFormat, source->componentType, source->planar, source->gpuMem);
    InputTransferred.bufferBytes = source->bufferBytes;
    CHECK_NVCV_ERROR(res);

    res = NvCVImage_Init(&OutputToBeTransferred, destination->width, destination->height, destination->pitch, destination->pixels, destination->pixelFormat, destination->componentType, destination->planar, destination->gpuMem);
    OutputToBeTransferred.bufferBytes = destination->bufferBytes;
    CHECK_NVCV_ERROR(res);

    NeedToSet = true;
    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::CreateArtifactReductionEffect(std::string modelsDir)
{
    if (!std::filesystem::exists(modelsDir)) {
        nosEngine.LogE("Models directory %s is not exists", modelsDir.c_str());
        return NOS_RESULT_FAILED;
    }
    NvCV_Status res;
    res = NvVFX_CreateEffect(NVVFX_FX_ARTIFACT_REDUCTION, &AR_EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetString(AR_EffectHandle, NVVFX_MODEL_DIRECTORY, modelsDir.c_str());
    CHECK_NVCV_ERROR(res);

    res = NvVFX_CreateEffect(NVVFX_FX_SR_UPSCALE, &UpScale_EffectHandle);
    CHECK_NVCV_ERROR(res);

    //res = NvVFX_Load(UpScale_EffectHandle);
    //CHECK_NVCV_ERROR(res);

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::CreateSuperResolutionEffect(std::string modelsDir)
{
    if (!std::filesystem::exists(modelsDir)) {
        nosEngine.LogE("Models directory %s is not exists", modelsDir.c_str());
        return NOS_RESULT_FAILED;
    }
    NvCV_Status res;
    res = NvVFX_CreateEffect(NVVFX_FX_SUPER_RES, &SuperRes_EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetString(SuperRes_EffectHandle, NVVFX_MODEL_DIRECTORY, modelsDir.c_str());
    CHECK_NVCV_ERROR(res);
    
    //res = NvVFX_Load(SuperRes_EffectHandle);
    //CHECK_NVCV_ERROR(res);

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::CreateAIGreenScreenEffect(std::string modelsDir)
{
    if (!std::filesystem::exists(modelsDir)) {
        nosEngine.LogE("Models directory %s is not exists", modelsDir.c_str());
        return NOS_RESULT_FAILED;
    }
    NvCV_Status res;
    res = NvVFX_CreateEffect(NVVFX_FX_GREEN_SCREEN, &AIGreenScreen_EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetString(AIGreenScreen_EffectHandle, NVVFX_MODEL_DIRECTORY, modelsDir.c_str());
    CHECK_NVCV_ERROR(res);

    //res = NvVFX_Load(AIGreenScreen_EffectHandle);
    //CHECK_NVCV_ERROR(res);

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::RunArtifactReduction(NvCVImage* input, NvCVImage* output)
{
    if (AR_EffectHandle == NULL)
        return NOS_RESULT_FAILED;

    NvCV_Status res = NVCV_SUCCESS;
    CHECK_NVCV_ERROR(res);
    
    res = NvVFX_SetCudaStream(AR_EffectHandle, NVVFX_CUDA_STREAM, reinterpret_cast<CUstream>(stream));
    CHECK_NVCV_ERROR(res);

    nosCUDAError nosres = nosCUDA->QueryStream(stream);
    res = NvCVImage_Transfer(input, &InputTransferred, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    nosres = nosCUDA->QueryStream(stream);
    nosCUDA->WaitStream(stream);
    CHECK_NVCV_ERROR(res);

    if(NeedToSet){

        res = NvVFX_SetImage(AR_EffectHandle, NVVFX_INPUT_IMAGE, &InputTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(AR_EffectHandle, NVVFX_OUTPUT_IMAGE, &OutputToBeTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetU32(AR_EffectHandle, NVVFX_MODE, 1);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_Load(AR_EffectHandle);
        CHECK_NVCV_ERROR(res);
        NeedToSet = false;
    }

    res = NvVFX_Run(AR_EffectHandle, 0);

    res = NvCVImage_Transfer(&OutputToBeTransferred, output, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    nosCUDA->WaitStream(stream);
    CHECK_NVCV_ERROR(res);
    //nosCUDA->WaitStream(reinterpret_cast<nosCUDAStream>(stream));

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::RunSuperResolution(NvCVImage* input, NvCVImage* output)
{
    if (SuperRes_EffectHandle == NULL)
        return NOS_RESULT_FAILED;

    NvCV_Status res = NVCV_SUCCESS;
    //res = NvVFX_CudaStreamCreate(&stream);
    //CHECK_NVCV_ERROR(res);

    res = NvVFX_SetCudaStream(SuperRes_EffectHandle, NVVFX_CUDA_STREAM, reinterpret_cast<CUstream>(stream));
    CHECK_NVCV_ERROR(res);

    res = NvCVImage_Transfer(input, &InputTransferred, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    //nosCUDA->WaitStream(reinterpret_cast<nosCUDAStream>(stream));
    CHECK_NVCV_ERROR(res);


    if (NeedToSet) {
        res = NvVFX_SetImage(SuperRes_EffectHandle, NVVFX_INPUT_IMAGE, &InputTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(SuperRes_EffectHandle, NVVFX_OUTPUT_IMAGE, &OutputToBeTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetCudaStream(SuperRes_EffectHandle, NVVFX_CUDA_STREAM, reinterpret_cast<CUstream>(stream));
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetU32(SuperRes_EffectHandle, NVVFX_MODE, 0);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_Load(SuperRes_EffectHandle);
        CHECK_NVCV_ERROR(res);
        NeedToSet = false;
    }

    res = NvVFX_Run(SuperRes_EffectHandle, 0);
    CHECK_NVCV_ERROR(res);

    res = NvCVImage_Transfer(&OutputToBeTransferred, output, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    nosCUDA->WaitStream(stream);
    CHECK_NVCV_ERROR(res);

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::RunAIGreenScreenEffect(NvCVImage* input, NvCVImage* output)
{

    if (AIGreenScreen_EffectHandle == NULL)
        return NOS_RESULT_FAILED;

    NvCV_Status res = NVCV_SUCCESS;

    if (AIGS_StateObjectHandle != NULL) {
        res = NvVFX_ResetState(AIGreenScreen_EffectHandle, AIGS_StateObjectHandle);
    }

    res = NvVFX_SetCudaStream(AIGreenScreen_EffectHandle, NVVFX_CUDA_STREAM, reinterpret_cast<CUstream>(stream));
    CHECK_NVCV_ERROR(res);

    res = NvCVImage_Transfer(input, &InputTransferred, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    CHECK_NVCV_ERROR(res);
    nosCUDA->WaitStream(stream);

    if (NeedToSet) {
        res = NvVFX_AllocateState(AIGreenScreen_EffectHandle, &AIGS_StateObjectHandle);
        res = NvVFX_SetObject(AIGreenScreen_EffectHandle, NVVFX_STATE, &AIGS_StateObjectHandle);

        res = NvVFX_SetImage(AIGreenScreen_EffectHandle, NVVFX_INPUT_IMAGE, &InputTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(AIGreenScreen_EffectHandle, NVVFX_OUTPUT_IMAGE, &OutputToBeTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetCudaStream(AIGreenScreen_EffectHandle, NVVFX_CUDA_STREAM, reinterpret_cast<CUstream>(stream));
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetU32(AIGreenScreen_EffectHandle, NVVFX_MODE, 0); //--mode=(0|1)
        CHECK_NVCV_ERROR(res);                                         //Selects the mode in which to run the application :
                                                                       //0 selects the best quality.                  
        res = NvVFX_Load(AIGreenScreen_EffectHandle);                  //1 selects the fastest performance.
        CHECK_NVCV_ERROR(res);
        NeedToSet = false;
    }

    res = NvVFX_Run(AIGreenScreen_EffectHandle, 0);
    CHECK_NVCV_ERROR(res);

    res = NvCVImage_Transfer(&OutputToBeTransferred, output, 1.0f, reinterpret_cast<CUstream>(stream), &Temp);
    CHECK_NVCV_ERROR(res);
    nosCUDA->WaitStream(stream);
    return NOS_RESULT_SUCCESS;
}
