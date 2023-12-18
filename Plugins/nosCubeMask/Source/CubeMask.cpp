// Copyright Zero Density AS. All Lefts Reserved.

#include <type_traits>

#include <Nodos/PluginAPI.h>
#include <Nodos/PluginHelpers.hpp>
#include <nosVulkanSubsystem/Helpers.hpp>

#include <nosCommon.h>

#include <atomic>
#include <thread>

#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtx/euler_angles.hpp>

typedef u32 uint;

NOS_INIT();

nosVulkanSubsystem* vk = 0;

namespace nos
{
#undef near
#undef far

NOS_REGISTER_NAME(Wireframe);
NOS_REGISTER_NAME(Render);
NOS_REGISTER_NAME(Mask);
NOS_REGISTER_NAME(Track);
NOS_REGISTER_NAME(Position);
NOS_REGISTER_NAME(Rotation);
NOS_REGISTER_NAME(Scale);
NOS_REGISTER_NAME(MVP);
NOS_REGISTER_NAME(Coeff);

struct Vertex : glm::vec3
{
	using glm::vec3::vec3;
};

static_assert(sizeof(Vertex) == 12);

template <class T>
bool NotSame(T a, T b)
{
	if constexpr (std::is_floating_point_v<T>)
	{
		return glm::abs(a - b) > glm::epsilon<T>();
	}
	else
	{
		return a != b;
	}
}

template <class... T>
u32 MakeFlags(std::unordered_map<Name, void*> const& pins, Name var, u32 idx, T&&... tail)
{
	return MakeFlags(pins, var, idx) | MakeFlags(pins, tail...);
}

template <>
u32 MakeFlags<>(std::unordered_map<Name, void*> const& pins, Name var, u32 idx)
{
	auto val = GetPinValue<bool>(pins, var);
	return val ? (*val << idx) : 0;
}

static glm::mat4 Perspective(f32 fovx, f32 pixelAspectRatio, glm::vec2 sensorSize, glm::vec2 centerShift)
{
	if (glm::vec2(0) == sensorSize)
	{
		sensorSize = glm::vec2(1);
		centerShift = glm::vec2(0);
	}

	const f32 near = 0.1;
	const f32 far = 10000;
	const f32 X = 1.f / tanf(glm::radians(fovx * 0.5));
	const f32 Y = -X * (sensorSize.x / sensorSize.y) * pixelAspectRatio;
	const auto S = -centerShift / sensorSize;
	const f32 Z = far / (far - near);
	return glm::mat4(
		glm::vec4(X, 0, 0, 0), glm::vec4(0, Y, 0, 0), glm::vec4(S.x, S.y, Z, 1.0f), glm::vec4(0, 0, -near * Z, 0));
}

static glm::mat4 MakeTransform(glm::vec3 pos, glm::vec3 rot)
{
	rot = glm::radians(rot);
	auto mat = (glm::mat3)glm::eulerAngleZYX(rot.z, -rot.y, -rot.x);
	return glm::lookAtLH(pos, pos + mat[0], mat[2]);
}

template <class T>
void AddParam(std::vector<nosShaderBinding>& inputs, std::unordered_map<Name, void*> const& pins, Name name)
{
	if (auto val = GetPinValue<T>(pins, name))
	{
		inputs.push_back({.Name = name, .FixedSize = val});
	}
}

template <class T>
bool GetValue(std::unordered_map<Name, const nos::fb::Pin*>& pins, Name name, T& dst)
{
	if (auto pin = pins[name])
	{
		if (flatbuffers::IsFieldPresent(pin, nos::fb::Pin::VT_DATA))
		{
			dst = *(T*)pin->data()->Data();
			return true;
		}
	}
	return false;
}

template <class T>
bool GetValue(std::unordered_map<Name, const nos::fb::Pin*>& pins, Name name, std::function<void(T*)>&& cb)
{
	if (auto pin = pins[name])
	{
		if (flatbuffers::IsFieldPresent(pin, nos::fb::Pin::VT_DATA) && pin->data()->size())
		{
			cb((T*)pin->data()->Data());
			return true;
		}
	}
	return false;
}

struct CubeMask : PinMapping
{

	nosVertexData Verts = {};

	enum
	{
		CAPTURING = 1,
		SAVING = 2,
	};

	nos::fb::TTrack Track;

	glm::vec3 Scale = {};
	glm::vec3 Position = {};
	glm::vec3 Rotation = {};

	void GenerateVertices2(std::vector<Vertex>& vertices, std::vector<glm::uvec3>& indices)
	{
		vertices = std::vector{
			Vertex(0, 0, 0),
			Vertex(0, 0, 1),
			Vertex(0, 1, 0),
			Vertex(0, 1, 1),
			Vertex(1, 0, 0),
			Vertex(1, 0, 1),
			Vertex(1, 1, 0),
			Vertex(1, 1, 1),
		};

		indices = {
			glm::uvec3(1, 0, 3), glm::uvec3(0, 2, 3),
			glm::uvec3(0, 4, 2), glm::uvec3(4, 6, 2),
			glm::uvec3(4, 5, 6), glm::uvec3(5, 7, 6),
			glm::uvec3(5, 1, 7), glm::uvec3(1, 3, 7),
			glm::uvec3(2, 6, 3), glm::uvec3(6, 7, 3),
			glm::uvec3(1, 5, 0), glm::uvec3(5, 4, 0),
		};
	}

	void LoadVertices()
	{
		if (Verts.Buffer.Memory.PID)
		{
			vk->DestroyResource(&Verts.Buffer);
		}

		std::vector<Vertex> vertices;
		std::vector<glm::uvec3> indices;

		GenerateVertices2(vertices, indices);

		u32 vsz = vertices.size() * sizeof(vertices[0]);
		u32 isz = indices.size() * sizeof(indices[0]);
		Verts.Buffer.Info.Type = NOS_RESOURCE_TYPE_BUFFER;
		Verts.Buffer.Info.Buffer.Size = vsz + isz;
		Verts.Buffer.Info.Buffer.Usage = nosBufferUsage(NOS_BUFFER_USAGE_VERTEX_BUFFER | NOS_BUFFER_USAGE_INDEX_BUFFER);
		Verts.VertexOffset = 0;
		Verts.IndexOffset = vsz;
		Verts.IndexCount = indices.size() * 3;
		// Verts.mutate_depth_func(nos::app::DepthFunction::LESS);
		// Verts.mutate_depth_test(true);
		// Verts.mutate_depth_write(true);

		vk->CreateResource(&Verts.Buffer);
		u8* mapping = vk->Map(&Verts.Buffer);
		memcpy(mapping, vertices.data(), vsz);
		memcpy(mapping + vsz, indices.data(), isz);
	}

	void Load(nos::fb::Node const& node)
	{
		auto name2pin = PinMapping::Load(node);
		LoadVertices();
	}

	CubeMask()
	{
		nosBuffer val;
		switch (nosEngine.GetDefaultValueOfType(NOS_NAME_STATIC("nos.fb.Track"), &val))
		{
		case NOS_RESULT_SUCCESS: flatbuffers::GetRoot<fb::Track>(val.Data)->UnPackTo(&Track); break;
		default: break;
		}
	}

	~CubeMask() { vk->DestroyResource(&Verts.Buffer); }

	// Node graph event callbacks
	static nosResult CanCreateNode(const nosFbNode* node) { return NOS_RESULT_SUCCESS; }
	static void OnNodeCreated(const nosFbNode* node, void** ctx)
	{
		CubeMask* c = new CubeMask();
		c->Load(*node);
		*ctx = c;
	}
	static void OnNodeUpdated(void* ctx, const nosFbNode* updatedNode) {}
	static void OnNodeDeleted(void* ctx, nosUUID nodeId) { delete (CubeMask*)ctx; }
	static void OnPinValueChanged(void* ctx, nosName pinName, nosUUID pinId, nosBuffer value)
	{
		auto c = static_cast<CubeMask*>(ctx);
		nos::Name pinMzName = pinName;

#define CHECK_AND_SET(name)                                                                                            \
	if (NOS_NAME_STATIC(#name) == pinMzName)                                                                           \
	{                                                                                                                  \
		c->name = *(std::remove_reference_t<decltype(c->name)>*)(value.Data);                                          \
		return;                                                                                                        \
	}

#define CHECK_AND_SET_TABLE(name, type)                                                                                \
	if (NOS_NAME_STATIC(#name) == pinMzName)                                                                           \
	{                                                                                                                  \
		flatbuffers::GetRoot<type>(value.Data)->UnPackTo(&c->name);                                                    \
		return;                                                                                                        \
	}

		CHECK_AND_SET_TABLE(Track, nos::fb::Track)
		CHECK_AND_SET(Position)
		CHECK_AND_SET(Rotation)
	}

	static void OnPinConnected(void* ctx, nosName pinName, nosUUID connectedPin, nosUUID) {}
	static void OnPinDisconnected(void* ctx, nosName pinName) {}
	static void OnPinShowAsChanged(void* ctx, nosName pinName, nosFbShowAs showAs) {}
	static void OnPathCommand(void* ctx, const nosPathCommand* command) {}

	// Execution
	static nosResult ExecuteNode(void* ctx, const nosNodeExecuteArgs* inArgs)
	{
		auto c = (CubeMask*)ctx;

		auto args = GetPinValues(inArgs);

		nosResourceShareInfo ClearColor{};

		std::vector<nosDrawCall> calls(2, {.BindingCount = 1, .Vertices = c->Verts});

		nosRunPass2Params pass = {
			.Key = NOS_NAME_STATIC("CUBE_MASK_PASS"),
			.Output = nos::vkss::DeserializeTextureInfo(GetPinValue<void>(args, NSN_Mask)),
			.DrawCalls = calls.data(),
			.DrawCallCount = (u32)calls.size(),
			.Wireframe = *GetPinValue<bool>(args, NSN_Wireframe),
		};

		auto track = GetPinValue<fb::TTrack>(args, NSN_Track);

		glm::vec3 pos = reinterpret_cast<glm::vec3&>(track.location);
		glm::vec3 rot = reinterpret_cast<glm::vec3&>(track.rotation);
		auto& distortion = track.lens_distortion;
		auto view = MakeTransform(pos, rot);
		auto prj = Perspective(track.fov,
							   track.pixel_aspect_ratio,
							   reinterpret_cast<glm::vec2&>(track.sensor_size),
							   reinterpret_cast<const glm::vec2&>(distortion.center_shift()));
		// glm::vec3 msize = *pins.Get<glm::vec3>("Size");
		glm::vec3 mpos = *GetPinValue<glm::vec3>(args, NSN_Position);
		glm::vec3 mrot = glm::radians(*GetPinValue<glm::vec3>(args, NSN_Rotation));
		glm::vec3 mscale = glm::radians(*GetPinValue<glm::vec3>(args, NSN_Scale));

		c->Scale    = mscale;
		c->Position = mpos;
		c->Rotation = mrot;
		c->Track = std::move(track);

		glm::mat4 model = glm::eulerAngleZYX(mrot.z, mrot.y, mrot.x) * glm::mat4(100.f);
		glm::mat4 MVP1 = glm::scale(model, mscale);
		glm::mat4 MVP2 = glm::scale(model, mscale * 1.5f);
		MVP2[3] = MVP1[3] = glm::vec4(mpos, 1.f);
		MVP1 = prj * view * MVP1;
		MVP2 = prj * view * MVP2;
		f32 one = 1.f, half = .5f;

		std::vector<std::vector<nosShaderBinding>> inputs = {
			{nos::vkss::ShaderBinding(NSN_MVP, MVP1), nos::vkss::ShaderBinding(NSN_Coeff, one)},
			{nos::vkss::ShaderBinding(NSN_MVP, MVP2), nos::vkss::ShaderBinding(NSN_Coeff, half)},
		};

		calls[0].Bindings = inputs[0].data();
		calls[1].Bindings = inputs[1].data();
		vk->RunPass2(0, &pass);

		return NOS_RESULT_SUCCESS;
	}

	static nosResult CanCopy(void* ctx, nosCopyInfo* copyInfo) { return NOS_RESULT_SUCCESS; }
	static nosResult BeginCopyFrom(void* ctx, nosCopyInfo* cospyInfo) { return NOS_RESULT_SUCCESS; }
	static nosResult BeginCopyTo(void* ctx, nosCopyInfo* copyInfo) { return NOS_RESULT_SUCCESS; }
	static void EndCopyFrom(void* ctx, nosCopyInfo* copyInfo) {}
	static void EndCopyTo(void* ctx, nosCopyInfo* copyInfo) {}
	// Menu & key events
	static void OnMenuRequested(void* ctx, const nosContextMenuRequest* request) {}
	static void OnMenuCommand(void* ctx, nosUUID itemID, uint32_t cmd) {}
	static void OnKeyEvent(void* ctx, const nosKeyEvent* keyEvent) {}
};

} // namespace nos

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportNodeFunctions(size_t* outSize, nosNodeFunctions** outList)
{
	*outSize = 1;
	if (!outList)
		return NOS_RESULT_SUCCESS;

	using namespace nos;
	auto* funcs = outList[0];
	funcs->ClassName = NOS_NAME_STATIC("CubeMask");
	funcs->CanCreateNode = CubeMask::CanCreateNode;
	funcs->OnNodeCreated = CubeMask::OnNodeCreated;
	funcs->OnNodeUpdated = CubeMask::OnNodeUpdated;
	funcs->OnNodeDeleted = CubeMask::OnNodeDeleted;
	funcs->OnPinValueChanged = CubeMask::OnPinValueChanged;
	funcs->OnPinConnected = CubeMask::OnPinConnected;
	funcs->OnPinDisconnected = CubeMask::OnPinDisconnected;
	funcs->OnPinShowAsChanged = CubeMask::OnPinShowAsChanged;
	funcs->OnPathCommand = CubeMask::OnPathCommand;
	funcs->ExecuteNode = CubeMask::ExecuteNode;
	funcs->BeginCopyFrom = CubeMask::BeginCopyFrom;
	funcs->BeginCopyTo = CubeMask::BeginCopyTo;
	funcs->EndCopyFrom = CubeMask::EndCopyFrom;
	funcs->EndCopyTo = CubeMask::EndCopyTo;
	funcs->OnMenuRequested = CubeMask::OnMenuRequested;
	funcs->OnMenuCommand = CubeMask::OnMenuCommand;
	funcs->OnKeyEvent = CubeMask::OnKeyEvent;

	nosEngine.RequestSubsystem(NOS_NAME_STATIC(NOS_VULKAN_SUBSYSTEM_NAME), 1, 0, (void**)&vk);

	// std::string root = nosEngine.Context->RootFolderPath;
	// {
	// 	std::string v0 = root + "./Shaders/CubeMask.vert";
	// 	std::string v1 = root + "./Shaders/CubeMask.frag";

	// 	std::vector<nosShaderInfo> infos = {
	// 		{
	// 			.Key = NOS_NAME_STATIC("CUBE_MASK_VERT"),
	// 			.Source = {.Stage = NOS_SHADER_STAGE_VERT, .GLSLPath = v0.data()},
	// 		},
	// 		{
	// 			.Key = NOS_NAME_STATIC("CUBE_MASK_FRAG"),
	// 			.Source = {.Stage = NOS_SHADER_STAGE_FRAG, .GLSLPath = v1.data()},
	// 		},
	// 	};
	// 	vk->RegisterShaders(infos.size(), infos.data());
	// }

	// {
	// 	std::vector<nosPassInfo> infos = {
	// 		{
	// 			.Key = NOS_NAME_STATIC("CUBE_MASK_PASS"),
	// 			.Shader = NOS_NAME_STATIC("CUBE_MASK_FRAG"),
	// 			.VertexShader = NOS_NAME_STATIC("CUBE_MASK_VERT"),
	// 			.Blend = true,
	// 			.MultiSample = 1,
	// 		},
	// 	};
	// 	vk->RegisterPasses(infos.size(), infos.data());
	// }
	return NOS_RESULT_SUCCESS;
}
}
