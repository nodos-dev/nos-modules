// Copyright Nodos AS. All Rights Reserved.

#include <Nodos/Helpers.hpp>

// External
#include <stb_image.h>
#include <stb_image_write.h>

// Framework
#include <Args.h>
#include <Builtins_generated.h>
#include <AppService_generated.h>

// nosNodes
#include "../Shaders/SRGB2Linear.frag.spv.dat"

namespace nos::utilities
{

NOS_REGISTER_NAME(SRGB2Linear_Pass);
NOS_REGISTER_NAME(SRGB2Linear_Shader);
NOS_REGISTER_NAME(Path);
NOS_REGISTER_NAME(Out);
NOS_REGISTER_NAME_SPACED(Nos_Utilities_ReadImage, "nos.utilities.ReadImage")

static nosResult GetShaders(size_t* outCount, nosShaderInfo* outShaders)
{
    *outCount = 1;
    if (!outShaders)
        return NOS_RESULT_SUCCESS;

	outShaders[0] = {.Key=NSN_SRGB2Linear_Shader, .Source = { .SpirvBlob = {(void*)SRGB2Linear_frag_spv, sizeof(SRGB2Linear_frag_spv)}}};
    return NOS_RESULT_SUCCESS;
}

static nosResult GetPasses(size_t* count, nosPassInfo* passes)
{
	*count = 1;
	if (!passes)
		return NOS_RESULT_SUCCESS;

	*passes = nosPassInfo{
		.Key = NSN_SRGB2Linear_Pass,
		.Shader = NSN_SRGB2Linear_Shader,
		.Blend = 0,
		.MultiSample = 1,
	};

	return NOS_RESULT_SUCCESS;
}

static nosResult GetFunctions(size_t* count, nosName* names, nosPfnNodeFunctionExecute* fns)
{
    *count = 1;
    if(!names || !fns)
        return NOS_RESULT_SUCCESS;
    
    *names = NOS_NAME_STATIC("ReadImage_Load");
    *fns = [](void* ctx, const nosNodeExecuteArgs* nodeArgs, const nosNodeExecuteArgs* functionArgs)
    {
        auto values = GetPinValues(nodeArgs);
		auto ids = GetPinIds(nodeArgs);
		std::filesystem::path path = GetPinValue<const char>(values, NSN_Path);
		try
		{
			if (!std::filesystem::exists(path))
			{
				nosEngine.LogE("Read Image cannot load file %s", path.string().c_str());
				return;
			}
			nosResourceShareInfo out = DeserializeTextureInfo(GetPinValue<void>(values, NSN_Out));
			nosResourceShareInfo tmp = out;
			
			int w, h, n;
			u8* img = stbi_load(path.string().c_str(), &w, &h, &n, 4);
			nosEngine.ImageLoad(img, nosVec2u(w,h), NOS_FORMAT_R8G8B8A8_SRGB, &tmp);
			free(img);

			nosCmd cmd;
			nosEngine.Begin(&cmd);
			nosEngine.Copy(cmd, &tmp, &out, 0);
			nosEngine.End(cmd);
			nosEngine.Destroy(&tmp);

			flatbuffers::FlatBufferBuilder fbb;
			auto dirty = CreateAppEvent(fbb, app::CreatePinDirtied(fbb, &ids[NSN_Out]));
			nosEngine.EnqueueEvent(&dirty);
		}
		catch(const std::exception& e)
		{
			nosEngine.LogE("Error while loading image: %s", e.what());
		}
    };
    
    return NOS_RESULT_SUCCESS;
}


void RegisterReadImage(nosNodeFunctions* fn)
{
	fn->TypeName = NSN_Nos_Utilities_ReadImage;
	fn->GetFunctions = GetFunctions;
    fn->GetShaders = GetShaders;
    fn->GetPasses = GetPasses;
}

// void RegisterReadImage(nosNodeFunctions* fn)
// {
	// auto& actions = functions["nos.ReadImage"];

	// actions.NodeCreated = [](fb::Node const& node, Args& args, void** context) {
	// 	*context = new ReadImageContext(node);
	// };

	// actions.EntryPoint = [](nos::Args& args, void* context) mutable {
	// 	auto path = args.Get<char>("Path");
	// 	if (!path || strlen(path) == 0)
	// 		return false;

	// 	i32 width, height, channels;
	// 	auto* ctx = static_cast<ReadImageContext*>(context);
	// 	u8* img = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	// 	bool ret = !!img && ctx->Upload(img, width, height, args.GetBuffer("Out"));
	// 	if (!ret)
	// 	{
	// 		nosEngine.LogE("ReadImage: Failed to load image");
	// 		flatbuffers::FlatBufferBuilder fbb;
	// 		std::vector<flatbuffers::Offset<nos::fb::NodeStatusMessage>> messages{nos::fb::CreateNodeStatusMessageDirect(
	// 			fbb, "Failed to load image", nos::fb::NodeStatusMessageType::FAILURE)};
	// 		HandleEvent(CreateAppEvent(
	// 			fbb,
	// 			nos::CreatePartialNodeUpdateDirect(fbb, &ctx->NodeId, ClearFlags::NONE, 0, 0, 0, 0, 0, 0, &messages)));
	// 	}
	// 	if (img)
	// 		stbi_image_free(img);

	// 	return ret;
	// };

	// actions.NodeRemoved = [](void* ctx, nos::fb::UUID const& id) { delete static_cast<ReadImageContext*>(ctx); };
// }

} // namespace nos::utilities