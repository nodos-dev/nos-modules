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

static MzResult GetShaders(size_t* count, const char** names, MzBuffer* spirv)
{
    *count = 1;
    if (!names || !spirv)
        return MZ_RESULT_SUCCESS;

    *names = "Linear2SRGB";
    spirv->Data = (void*)Linear2SRGB_frag_spv;
    spirv->Size = sizeof(Linear2SRGB_frag_spv);
    return MZ_RESULT_SUCCESS;
}

static MzResult GetPasses(size_t* count, MzPassInfo* passes)
{
    *count = 1;
    if (!passes)
        return MZ_RESULT_SUCCESS;

    *passes = MzPassInfo{
        .Key = "Linear2SRGB_Pass",
        .Shader = "Linear2SRGB",
        .Blend = 0,
        .MultiSample = 1,
    };

    return MZ_RESULT_SUCCESS;
}

static MzResult GetFunctions(size_t* count, const char** names, PFN_NodeFunctionExecute* fns)
{
    *count = 1;
    if(!names || !fns)
        return MZ_RESULT_SUCCESS;

    *names = "WriteImage_Save";
    *fns = [](void* ctx, const MzNodeExecuteArgs* nodeArgs, const MzNodeExecuteArgs* functionArgs)
    {
        auto values = GetPinValues(nodeArgs);
        std::filesystem::path path = GetPinValue<const char>(values, "Path");
        if (std::filesystem::is_directory(path.parent_path()))
        {
            mzEngine.LogE("Write Image cannot write to directory %s", path.parent_path().c_str());
            return;
        }
        MzResourceShareInfo input = DeserializeTextureInfo(GetPinValue<void>(values, "Input"));
        MzResourceShareInfo srgb = input;
        srgb.info.texture.format = MZ_FORMAT_R8G8B8A8_SRGB;
        srgb.info.texture.usage = MZ_IMAGE_USAGE_NONE;
        mzEngine.Create(&srgb);
        MzResourceShareInfo buf = {};
        MzCmd cmd;
        mzEngine.Begin(&cmd);
        mzEngine.Blit(cmd, &input, &srgb);
        mzEngine.Download(cmd, &srgb, &buf);
        mzEngine.End(cmd);

        if (auto buf2write = mzEngine.Map(&buf))
            if (!stbi_write_png(path.string().c_str(), input.info.texture.width, input.info.texture.height, 4, buf2write, input.info.texture.width * 4))
                mzEngine.LogE("WriteImage: Unable to write frame to file", "");

        mzEngine.Destroy(&buf);
        mzEngine.Destroy(&srgb);
    };
    
    return MZ_RESULT_SUCCESS;
}

void RegisterWriteImage(MzNodeFunctions* fn)
{
    *fn = {
        .TypeName = "mz.utilities.WriteImage",
        .GetFunctions = GetFunctions,
        .GetShaders = GetShaders,
        .GetPasses = GetPasses,
    };
}

} // namespace mz