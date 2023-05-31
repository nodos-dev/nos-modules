// Copyright MediaZ AS. All Rights Reserved.

#include <MediaZ/Helpers.hpp>

// External
#include <stb_image.h>
#include <stb_image_write.h>

// Framework
#include <Args.h>
#include <Builtins_generated.h>
#include <AppService_generated.h>

// mzNodes
#include "../Shaders/SRGB2Linear.frag.spv.dat"

namespace mz::utilities
{

static MzResult GetShaders(size_t* count, const char** names, MzBuffer* spirv)
{
	*count = 1;
	if (!names || !spirv)
		return MZ_RESULT_SUCCESS;

	*names = "ReadImage_SRGB2Linear";
	spirv->Data = (void*)SRGB2Linear_frag_spv;
	spirv->Size = sizeof(SRGB2Linear_frag_spv);
	return MZ_RESULT_SUCCESS;
}

static MzResult GetPasses(size_t* count, MzPassInfo* passes)
{
	*count = 1;
	if (!passes)
		return MZ_RESULT_SUCCESS;

	*passes = MzPassInfo{
		.Key = "SRGB2Linear_Pass",
		.Shader = "ReadImage_SRGB2Linear",
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
    
    *names = "ReadImage_Load";
    *fns = [](void* ctx, const MzNodeExecuteArgs* nodeArgs, const MzNodeExecuteArgs* functionArgs)
    {
        auto values = GetPinValues(nodeArgs);
        std::filesystem::path path = GetPinValue<const char>(values, "Path");
        if (std::filesystem::exists(path))
        {
            mzEngine.LogE("Read Image cannot load file %s", path.string().c_str());
            return;
        }
        MzResourceShareInfo out = DeserializeTextureInfo(GetPinValue<void>(values, "Out"));
        MzResourceShareInfo tmp = out;
		
		int w, h, n;
		u8* img = stbi_load(path.string().c_str(), &w, &h, &n, 4);
		mzEngine.ImageLoad(img, MzVec2u(w,h), MZ_FORMAT_R8G8B8A8_SRGB, &tmp);
		free(img);

        MzCmd cmd;
        mzEngine.Begin(&cmd);
        mzEngine.Copy(cmd, &tmp, &out, 0);
        mzEngine.End(cmd);
        mzEngine.Destroy(&tmp);
    };
    
    return MZ_RESULT_SUCCESS;
}


void RegisterReadImage(MzNodeFunctions* fn)
{
    *fn = {
        .TypeName = "mz.utilities.ReadImage",
        .GetFunctions = GetFunctions,
        .GetShaders = GetShaders,
        .GetPasses = GetPasses,
    };
}

// void RegisterReadImage(MzNodeFunctions* fn)
// {
	// auto& actions = functions["mz.ReadImage"];

	// actions.NodeCreated = [](fb::Node const& node, Args& args, void** context) {
	// 	*context = new ReadImageContext(node);
	// };

	// actions.EntryPoint = [](mz::Args& args, void* context) mutable {
	// 	auto path = args.Get<char>("Path");
	// 	if (!path || strlen(path) == 0)
	// 		return false;

	// 	i32 width, height, channels;
	// 	auto* ctx = static_cast<ReadImageContext*>(context);
	// 	u8* img = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	// 	bool ret = !!img && ctx->Upload(img, width, height, args.GetBuffer("Out"));
	// 	if (!ret)
	// 	{
	// 		mzEngine.LogE("ReadImage: Failed to load image");
	// 		flatbuffers::FlatBufferBuilder fbb;
	// 		std::vector<flatbuffers::Offset<mz::fb::NodeStatusMessage>> messages{mz::fb::CreateNodeStatusMessageDirect(
	// 			fbb, "Failed to load image", mz::fb::NodeStatusMessageType::FAILURE)};
	// 		mzEngine.HandleEvent(CreateAppEvent(
	// 			fbb,
	// 			mz::CreatePartialNodeUpdateDirect(fbb, &ctx->NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &messages)));
	// 	}
	// 	if (img)
	// 		stbi_image_free(img);

	// 	return ret;
	// };

	// actions.NodeRemoved = [](void* ctx, mz::fb::UUID const& id) { delete static_cast<ReadImageContext*>(ctx); };
// }

} // namespace mz::utilities