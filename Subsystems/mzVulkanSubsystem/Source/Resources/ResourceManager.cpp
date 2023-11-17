#include "ResourceManager.h"

#include <mzVulkan/Device.h>
#include <mzVulkan/Command.h>
#include <mzVulkan/Buffer.h>

#include "Globals.h"
#include "Embed/Test.png.dat"
#include "Embed/unlic.png.dat"

// External
#include <stb_image.h>

// MediaZ SDK
#include <glm/common.hpp>

namespace mz::vkss
{

void ResourceManager::Clear()
{
	std::scoped_lock guard(Mutex);
	Colors.clear();
	SrcStockTextures.clear();
	ResizedStockTextures.clear();
	ImagePool = {};
	BufferPool = {};
}

rc<vk::Image> ResourceManager::GetColorTexture(glm::vec4 inColor)
{
	std::scoped_lock guard(Mutex);
	glm::vec4 colorf = glm::clamp(inColor, 0.f, 1.f);
	u32 color = (u32)(colorf.x * 255) | ((u32)(colorf.y * 255) << 8) | ((u32)(colorf.z * 255) << 16) | (
		            (u32)(colorf.w * 255) << 24);
	auto& img = Colors[color];
	if (!img)
	{
		std::string tag(256, 0);
		snprintf(tag.data(), 256, "Color 0x%X", color);
		tag.shrink_to_fit();
		img = Create<vk::Image>({
			.Extent = {4, 4},
			.Format = VK_FORMAT_R8G8B8A8_UNORM,
			.Usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		}, std::move(tag));
		auto cmd = GVkDevice->GetPool()->BeginCmd();
		img->Clear(cmd, {.float32 = {colorf.x, colorf.y, colorf.z, colorf.w}});
		cmd->Submit()->Wait();
	}
	return img;
}

vk::rc<vk::Image> ResourceManager::GetStockTexture(u32 width,
                                               u32 height,
                                               mz::fb::Format fmt,
                                               mz::fb::StockTexture stockType)
{
	std::scoped_lock guard(Mutex);
	const u64 hash = (u64)width | ((u64)height << 16) | ((u64)fmt << 32) | ((u64)stockType << 48);
	auto& sig = ResizedStockTextures[hash];
	if (!sig)
	{
		sig = Create<vk::Image>({
			.Extent = {width, height},
			.Format = (VkFormat)fmt,
			.Usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			         VK_IMAGE_USAGE_SAMPLED_BIT}, "Stock Texture");
		auto cmd = GVkDevice->GetPool()->BeginCmd();
		sig->BlitFrom(cmd, SrcStockTextures[stockType], VK_FILTER_LINEAR);
		cmd->Submit()->Wait();
	}
	return sig;
}

vk::rc<vk::Image> ResourceManager::GetStockTextureDirect(mz::fb::StockTexture stockType)
{
	std::scoped_lock guard(Mutex);
	return SrcStockTextures[stockType];
}

void ResourceManager::LoadStockTextures()
{
	std::scoped_lock guard(Mutex);
	using mz::fb::StockTexture;
	u8* noVideoImgData;
	u8* logoImgData;

	auto cmd = GVkDevice->GetPool()->BeginCmd();
	auto loadImageFromMemory = [this, cmd](const u8* src, size_t srcSize, StockTexture type, u8*& dst) {
		i32 x, y, n;
		dst = stbi_load_from_memory(src, srcSize, &x, &y, &n, 4);
		if (!dst)
		{
			x = y = 1;
			const u8 yellow[4] = {255, 255, 0, 255};
			dst = (u8*)malloc(4);
			memcpy(dst, yellow, 4);
		}
		auto srcTestSignal = SrcStockTextures[type] = Create<vk::Image>( {.Extent = {(u32)x, (u32)y},
			 .Format = VK_FORMAT_R8G8B8A8_SRGB,
			 .Usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT});

		srcTestSignal->Upload(
			cmd,
			vk::Buffer::New(
				GVkDevice,
				vk::BufferCreateInfo{.Size = x * y * 4u, .Usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT, .Data = dst}));
	};

	loadImageFromMemory(Test_png, sizeof(Test_png), StockTexture::MZNOVIDEO, noVideoImgData);
	loadImageFromMemory(unlic_png, sizeof(unlic_png), StockTexture::MZLOGO, logoImgData);

	// load other stock textures
	SrcStockTextures[StockTexture::BLACK] = GetColorTexture(glm::vec4(0, 0, 0, 1));
	SrcStockTextures[StockTexture::WHITE] = GetColorTexture(glm::vec4(1, 1, 1, 1));

	cmd->Submit()->Wait();
	free(noVideoImgData);
	free(logoImgData);
}

// TODO: Port
// void ResourceManager::GetStatistics(editor::TEngineMetrics& metrics)
// {
// 	std::scoped_lock guard(Mutex);
// 	metrics.available_resource_count = ImagePool.GetAvailableResourceCount() + BufferPool.GetAvailableResourceCount();
// 	metrics.used_resource_count = ImagePool.GetUsedResourceCount() + BufferPool.GetUsedResourceCount();
// 	metrics.imported_resource_count = ImportedResources.size();
// 	metrics.memory_used_for_resources_k = (ImagePool.GetTotalMemoryUsage() + BufferPool.GetTotalMemoryUsage()) / 1000;
// }
//
// void ResourceManager::GetResourceInfos(editor::TResources& resources)
// {
// 	std::vector<editor::TResourceInfo> infos;
// 	ImagePool.GetResourceInfos(infos);
// 	BufferPool.GetResourceInfos(infos);
// 	for (auto& [handle, info] : ImportedResources)
// 	{
// 		infos.push_back(editor::TResourceInfo{
// 			.handle = reinterpret_cast<uint64_t>(handle),
// 			.imported = true,
// 			.memory_used = 0,
// 			.used = true,
// 			.tag = info.Tag,
// 		});
// 	}
// 	for (auto& info : infos)
// 		resources.resources.emplace_back(std::make_unique<editor::TResourceInfo>(info));
// }

} // namespace mz::engine
