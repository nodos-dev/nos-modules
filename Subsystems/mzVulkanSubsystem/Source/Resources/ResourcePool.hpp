#pragma once

// Subsystem
#include "Globals.h"

// MediaZ SDK
#include <MediaZ/Types.h>

// Vulkan base
#include <mzVulkan/Common.h>
#include <mzVulkan/Image.h>

// std
#include <unordered_map>

namespace mz::vkss
{

template <typename ResourceT, typename CreationInfoT,
          typename CreationInfoHasherT = std::hash<CreationInfoT>,
          typename CreationInfoEqualsT = std::equal_to<CreationInfoT>,
		  size_t FreeSlotsPerCreationInfo = 12> // TODO: A more sophisticated strategy might be implemented
class ResourcePool
{
public:
	rc<ResourceT> Get(CreationInfoT const& info, std::string tag)
	{
		auto freeIt = Free.find(info);
		if (freeIt == Free.end())
		{
			auto res = typename ResourceT::New(GVkDevice, info);
			Used[res.get()] = {tag, info, res};
			return res;
		}
		auto& resources = freeIt->second;
		auto res = resources.back();
		resources.pop_back();
		if (resources.empty())
			Free.erase(freeIt);
		Used[res.get()] = {tag, info, res};
		return res;
	}

	bool Release(ResourceT* resource)
	{
		auto usedIt = Used.find(resource);
		if (usedIt == Used.end())
		{
			mzEngine.LogW("ResourcePool: %d is already released", resource);
			return false;
		}
		auto [tag, info, res] = usedIt->second;
		Used.erase(usedIt);
		auto& freeList = Free[info];
		if (freeList.size() < FreeSlotsPerCreationInfo)
			Free[info].push_back(std::move(res));
		return true;
	}

	void Purge()
	{
		Free.clear();
	}

	bool IsUsed(ResourceT* resource) const
	{
		return Used.contains(resource);
	}

	uint64_t GetAvailableResourceCount() const
	{
		uint64_t ret = 0;
		for (auto& [info, freeList] : Free)
			ret += freeList.size();
		return ret;
	}

	uint64_t GetUsedResourceCount() const { return Used.size(); }

	uint64_t GetTotalMemoryUsage() const { return GetAvailableResourceMemoryUsage() + GetUsedResourceMemoryUsage();	}

	uint64_t GetAvailableResourceMemoryUsage() const
	{
		uint64_t ret = 0;
		for (auto& [info, freeList] : Free)
			for (auto& free : freeList)
				ret += free->Allocation.GetSize();
		return ret;
	}

	uint64_t GetUsedResourceMemoryUsage() const
	{
		uint64_t ret = 0;
		for (auto& [handle, info] : Used)
			ret += info.Resource->Allocation.GetSize();
		return ret;
	}

	struct UsedResourceInfo
	{
		std::string Tag;
		CreationInfoT CreationInfo;
		rc<ResourceT> Resource;
	};

	// void GetResourceInfos(std::vector<editor::TResourceInfo>& out)
	// {
	// 	for (auto& [handle, usedInfo] : Used)
	// 	{
	// 		editor::TResourceInfo info;
	// 		out.emplace_back(editor::TResourceInfo{
	// 			.handle = reinterpret_cast<uint64_t>(handle),
	// 			.imported = false,
	// 			.memory_used = usedInfo.Resource->Allocation.GetSize(),
	// 			.used = true,
	// 			.tag = usedInfo.Tag,
	// 		});
	// 	}
	// 	for (auto& [creationInfo, freeList] : Free)
	// 		for (auto& res : freeList)
	// 		{
	// 			out.emplace_back(editor::TResourceInfo{
	// 				.handle = reinterpret_cast<uint64_t>(res.get()),
	// 				.imported = false,
	// 				.memory_used = res->Allocation.GetSize(),
	// 				.used = false,
	// 				.tag = "Free Resource",
	// 			});
	// 		}
	// }

protected:
	std::unordered_map<ResourceT*, UsedResourceInfo> Used;
	std::unordered_map<CreationInfoT, std::vector<rc<ResourceT>>, CreationInfoHasherT, CreationInfoEqualsT> Free;
};

// TODO: Move below to ResourceManager.cpp after if/when ResourcePool is fully generic
namespace detail
{
// Since some of the fields of createinfo structs are not used, hashers and equality checks are implemented here instead of mzVulkan.
struct ImageCreateInfoHasher
{
	size_t operator()(vk::ImageCreateInfo const& info) const
	{
		size_t result = 0;
		vk::hash_combine(result, info.Extent.width, info.Extent.height, info.Format, info.Usage, info.Samples, info.Tiling, info.Flags, info.Type);
		return result;
	}
};

struct ImageCreateInfoEquals
{
	bool operator()(vk::ImageCreateInfo const& l, vk::ImageCreateInfo const& r) const
	{
		return l.Extent == r.Extent && l.Format == r.Format && l.Usage == r.Usage && l.Samples == r.
			   Samples && l.Tiling == r.Tiling && l.Flags == r.Flags && l.Type == r.Type;
	}
};

struct BufferCreateInfoHasher
{
	size_t operator()(vk::BufferCreateInfo const& info) const
	{
		size_t result = 0;
		vk::hash_combine(result, info.Size, info.Mapped, info.VRAM, info.Usage, info.Type);
		return result;
	}
};

struct BufferCreateInfoEquals
{
	bool operator()(vk::BufferCreateInfo const& l, vk::BufferCreateInfo const& r) const
	{
		return l.Size == r.Size && l.Mapped == r.Mapped && l.VRAM == r.VRAM && l.Usage == r.Usage && l.Type == r.Type;
	}
};
} // namespace detail

using ImagePool = ResourcePool<vk::Image, vk::ImageCreateInfo, detail::ImageCreateInfoHasher, detail::ImageCreateInfoEquals>;
using BufferPool = ResourcePool<vk::Buffer, vk::BufferCreateInfo, detail::BufferCreateInfoHasher, detail::BufferCreateInfoEquals>;

}
