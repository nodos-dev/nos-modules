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
        path = std::filesystem::canonical(path);
        if (!std::filesystem::is_directory(path.parent_path()))
        {
            mzEngine.LogE("Write Image cannot write to directory %s", path.parent_path().string().c_str());
            return;
        }
		mzResourceShareInfo input = DeserializeTextureInfo(values[MZN_In]);
        mzResourceShareInfo srgb = input;
        srgb.Info.Texture.Format = MZ_FORMAT_R8G8B8A8_SRGB;
        srgb.Info.Texture.Usage = MZ_IMAGE_USAGE_NONE;
        mzEngine.Create(&srgb);
        mzResourceShareInfo buf = {};
        mzCmd cmd;
        mzEngine.Begin(&cmd);
        mzEngine.Blit(cmd, &input, &srgb);
        mzEngine.Download(cmd, &srgb, &buf);
        mzEngine.End(cmd);

        if (auto buf2write = mzEngine.Map(&buf))
            if (!stbi_write_png(path.string().c_str(), input.Info.Texture.Width, input.Info.Texture.Height, 4, buf2write, input.Info.Texture.Width * 4))
                mzEngine.LogE("WriteImage: Unable to write frame to file", "");

        mzEngine.Destroy(&buf);
        mzEngine.Destroy(&srgb);
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