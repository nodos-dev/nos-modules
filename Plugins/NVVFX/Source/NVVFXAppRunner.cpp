#include "NVVFXAppRunner.h"

NVVFXAppRunner::NVVFXAppRunner() : AR_EffectHandle(NULL)
{
}

NVVFXAppRunner::~NVVFXAppRunner()
{
    if (InputTransferred.pixels != nullptr)
        NvCVImage_Destroy(&InputTransferred);

    if (Temp.pixels != nullptr)
        NvCVImage_Destroy(&Temp);

    if (OutputToBeTransferred.pixels != nullptr)
        NvCVImage_Destroy(&OutputToBeTransferred);
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
    return nosResult();
}

nosResult NVVFXAppRunner::CreateUpscaleEffect(std::string modelsDir)
{
    if (!std::filesystem::exists(modelsDir)) {
        nosEngine.LogE("Models directory %s is not exists", modelsDir.c_str());
        return NOS_RESULT_FAILED;
    }
    NvCV_Status res;
    res = NvVFX_CreateEffect(NVVFX_FX_ARTIFACT_REDUCTION, &AR_EffectHandle);
    CHECK_NVCV_ERROR(res);

    res = NvVFX_SetString(AR_EffectHandle, NVVFX_MODEL_DIRECTORY, "C:/WorkInParallel/MAXINE-VFX-SDK/models");
    CHECK_NVCV_ERROR(res);

    res = NvVFX_CreateEffect(NVVFX_FX_SR_UPSCALE, &UpScale_EffectHandle);
    CHECK_NVCV_ERROR(res);

    return NOS_RESULT_SUCCESS;
}

nosResult NVVFXAppRunner::Run(NvCVImage* input, NvCVImage* output)
{
    if (AR_EffectHandle == NULL)
        return NOS_RESULT_FAILED;

    NvCV_Status res = NVCV_SUCCESS;
    CUstream stream = 0;
    //res = NvVFX_CudaStreamCreate(&stream);
    CHECK_NVCV_ERROR(res);

    bool needTransfer = true;

    if (false) {
        /*if(InputTransferred.pixels != nullptr)
            NvCVImage_Destroy(&InputTransferred);
        
        if(Temp.pixels != nullptr)
          res  NvCVImage_Destroy(&Temp);
        
        if(OutputToBeTransferred.pixels != nullptr)
            NvCVImage_Destroy(&OutputToBeTransferred);*/

        if (needTransfer) {
            NvCVImage dummy1 = {};
            NvCVImage dummy2 = {};
            res = NvCVImage_Alloc(&dummy1, input->width, input->height, NVCV_BGRA, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 1);
            CHECK_NVCV_ERROR(res);

            //res = NvCVImage_Alloc(&Temp, input->width, input->height, NVCV_RGBA, NVCV_F32, NVCV_CHUNKY, NVCV_CUDA, 1);
            //CHECK_NVCV_ERROR(res);

            res = NvCVImage_Alloc(&dummy2, output->width, output->height, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_CUDA, 32);
            CHECK_NVCV_ERROR(res);
        }
        else {
            res = NvCVImage_Alloc(&InputTransferred, input->width, input->height, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CUDA, 32);
            CHECK_NVCV_ERROR(res);

            res = NvCVImage_Alloc(&Temp, input->width, input->height, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CUDA, 32);
            CHECK_NVCV_ERROR(res);

            res = NvCVImage_Alloc(&OutputToBeTransferred, output->width, output->height, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CUDA, 32);
            CHECK_NVCV_ERROR(res);
        }


        LastWidth = input->width;
        LastHeight = input->height;
        LastComponentType = input->componentType;
        LastPixelFormat = input->pixelFormat;
    }

    //float* data1 = new float[input->bufferBytes/4];
    //cudaMemcpy(data1, input->pixels, input->bufferBytes, cudaMemcpyDeviceToHost);
    if (needTransfer) {
        //res = NvVFX_SetCudaStream(AR_EffectHandle, NVVFX_CUDA_STREAM, stream);
        CHECK_NVCV_ERROR(res);

        int temp = input->height;
        input->height = temp + 72;
        output->height = temp + 72;

        InputTransferred.height = temp + 72;
        OutputToBeTransferred.height = temp + 72;

        res = NvCVImage_Transfer(input, &InputTransferred, 1.0f, stream, &Temp);
        cudaStreamSynchronize(stream);
        CHECK_NVCV_ERROR(res);

        InputTransferred.height = temp ;
        OutputToBeTransferred.height = temp ;
        //
        //res = NvCVImage_Transfer(&InputTransferred, &OutputToBeTransferred, 1.0f, stream, &Temp);
        //cudaStreamSynchronize(stream);
        //CHECK_NVCV_ERROR(res);

        //float* data2 = new float[InputTransferred.bufferBytes/4];
        //cudaMemcpy(data2, InputTransferred.pixels, InputTransferred.bufferBytes, cudaMemcpyDeviceToHost);


        res = NvVFX_SetImage(AR_EffectHandle, NVVFX_INPUT_IMAGE, &InputTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(AR_EffectHandle, NVVFX_OUTPUT_IMAGE, &OutputToBeTransferred);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetCudaStream(AR_EffectHandle, NVVFX_CUDA_STREAM, stream);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetU32(AR_EffectHandle, NVVFX_MODE, 1);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_Load(AR_EffectHandle);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_Run(AR_EffectHandle, 0);
        CHECK_NVCV_ERROR(res);

        InputTransferred.height = temp + 72;
        OutputToBeTransferred.height = temp + 72;

        res = NvCVImage_Transfer(&OutputToBeTransferred, output, 1.0f, stream, &Temp);
        cudaStreamSynchronize(stream);
        CHECK_NVCV_ERROR(res);

        input->height = temp;
        output->height = temp;

        InputTransferred.height = temp;
        OutputToBeTransferred.height = temp;
        //float* data3 = new float[OutputToBeTransferred.bufferBytes/4];
        //cudaMemcpy(data3, OutputToBeTransferred.pixels, OutputToBeTransferred.bufferBytes, cudaMemcpyDeviceToHost);

       // res = NvCVImage_Transfer(input, output, 1.0f, stream, &Temp);



        //input->pitch = temp * 2;
        //input->componentBytes = 4;
        //input->componentType = NVCV_F32;

        cudaStreamSynchronize(stream);
        CHECK_NVCV_ERROR(res);

//        input->pitch = temp;

        //float* data4 = new float[output->bufferBytes/4];
        //cudaMemcpy(data4, output->pixels, output->bufferBytes, cudaMemcpyDeviceToHost);

        static int a = 0;
        if (a++ > 5) {
            int devughere = 0;
        }

    }
    else {
        res = NvVFX_SetCudaStream(UpScale_EffectHandle, NVVFX_CUDA_STREAM, stream);
        CHECK_NVCV_ERROR(res);

        //res = NvCVImage_Transfer(input, &InputTransferred, 1.0f, stream, &Temp);
        //cudaStreamSynchronize(stream);
        //CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(UpScale_EffectHandle, NVVFX_INPUT_IMAGE, input);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetImage(UpScale_EffectHandle, NVVFX_OUTPUT_IMAGE, output);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_SetCudaStream(UpScale_EffectHandle, NVVFX_CUDA_STREAM, stream);
        CHECK_NVCV_ERROR(res);

        //res = NvVFX_SetF32(UpScale_EffectHandle, NVVFX_STRENGTH, 1.0f);
        //CHECK_NVCV_ERROR(res);

        /*res = NvVFX_SetU32(UpScale_EffectHandle, NVVFX_MODE, 1);
        CHECK_NVCV_ERROR(res);*/

        res = NvVFX_Load(UpScale_EffectHandle);
        CHECK_NVCV_ERROR(res);

        res = NvVFX_Run(UpScale_EffectHandle, 0);
        CHECK_NVCV_ERROR(res);

        //res = NvCVImage_Transfer(&OutputToBeTransferred, output, 255.0f, stream, &Temp);
        //cudaStreamSynchronize(stream);
        //CHECK_NVCV_ERROR(res);
    }

    //uint8_t* charData = new uint8_t[output->bufferBytes];
    //cudaMemcpy(charData, output->pixels, output->bufferBytes, cudaMemcpyDeviceToHost);


    //res = NvVFX_Load(UpScale_EffectHandle);
    //CHECK_NVCV_ERROR(res);

    //res = NvVFX_Run(AR_EffectHandle, 0);
    //CHECK_NVCV_ERROR(res);


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

bool NVVFXAppRunner::ShouldAllocateBuffer(NvCVImage* in)
{
    return in->width != LastWidth || in->height != LastHeight || LastComponentType != in->componentType || LastPixelFormat != in->pixelFormat;
}
