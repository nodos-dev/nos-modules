// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>
#include "Builtins_generated.h"
#include <AppService_generated.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include "../Shaders/Linear2SRGB.frag.spv.dat"
// #include "Filters/Sampler.frag.spv.dat"

namespace mz::utilities
{
MZ_REGISTER_NAME(Linear2SRGB_Pass);
MZ_REGISTER_NAME(Linear2SRGB_Shader);
MZ_REGISTER_NAME(Path);
MZ_REGISTER_NAME(In);
MZ_REGISTER_NAME_SPACED(Mz_Utilities_WriteImage, "mz.utilities.WriteImage")

static mzResult GetShaders(size_t* count, mzName* names, mzBuffer* spirv)
{
    *count = 1;
    if (!names || !spirv)
        return MZ_RESULT_SUCCESS;

    *names = MZN_Linear2SRGB_Shader;
    spirv->Data = (void*)Linear2SRGB_frag_spv;
    spirv->Size = sizeof(Linear2SRGB_frag_spv);
    return MZ_RESULT_SUCCESS;
}

static mzResult GetPasses(size_t* count, mzPassInfo* passes)
{
    *count = 1;
    if (!passes)
        return MZ_RESULT_SUCCESS;

    *passes = mzPassInfo{
		.Key = MZN_Linear2SRGB_Pass,
		.Shader = MZN_Linear2SRGB_Shader,
        .Blend = 0,
        .MultiSample = 1,
    };

    return MZ_RESULT_SUCCESS;
}

static mzResult GetFunctions(size_t* count, mzName* names, mzPfnNodeFunctionExecute* fns)
{

    *count = 1;
    if(!names || !fns)
        return MZ_RESULT_SUCCESS;

    *names = MZ_NAME_STATIC("WriteImage_Save");
    *fns = [](void* ctx, const mzNodeExecuteArgs* nodeArgs, const mzNodeExecuteArgs* functionArgs)
    {
        auto values = GetPinValues(nodeArgs);
		std::filesystem::path path = GetPinValue<const char>(values, MZN_Path);
        
        if (!std::filesystem::is_directory(path.parent_path()))
        {
            mzEngine.LogE("Write Image cannot write to directory %s", path.parent_path().string().c_str());
            return;
        }
		mzResourceShareInfo input = DeserializeTextureInfo(values[MZN_In]);

        struct Captures
        {
            mzResourceShareInfo srgb;
            mzResourceShareInfo buf = {};
            std::filesystem::path path;
        }* captures = new Captures{.srgb=input,.path=std::move(path)};
      
        captures->srgb.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
        captures->srgb.Info.Texture.Usage = mzImageUsage(MZ_IMAGE_USAGE_TRANSFER_SRC | MZ_IMAGE_USAGE_TRANSFER_DST);
        mzEngine.Create(&captures->srgb);

        mzEngine.Blit(0, &input, &captures->srgb);
        mzEngine.Download(0, &captures->srgb, &captures->buf);

        mzEngine.RegisterCommandFinishedCallback(0, captures, [](void* data) {
            auto captures = (Captures*)data;
            if (auto buf2write = mzEngine.Map(&captures->buf))
                if (!stbi_write_png(captures->path.string().c_str(), captures->srgb.Info.Texture.Width, captures->srgb.Info.Texture.Height, 4, buf2write, captures->srgb.Info.Texture.Width * 4))
                    mzEngine.LogE("WriteImage: Unable to write frame to file", "");
            mzEngine.Destroy(&captures->buf);
            mzEngine.Destroy(&captures->srgb);
            delete captures;
        });

    };
    
    return MZ_RESULT_SUCCESS;
}

void RegisterWriteImage(mzNodeFunctions* fn)
{
    *fn = {
		.TypeName = MZN_Mz_Utilities_WriteImage,
        .GetFunctions = GetFunctions,
        .GetShaders = GetShaders,
        .GetPasses = GetPasses,
    };
}

} // namespace mz