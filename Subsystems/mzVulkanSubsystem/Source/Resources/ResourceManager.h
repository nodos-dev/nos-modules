#pragma once

// Subsystem
#include "ResourcePool.hpp"

// Vulkan base
#include <mzVulkan/Common.h>
#include <mzVulkan/Image.h>

// stl
#include <unordered_map>

// MediaZ SDK
#include <glm/vec4.hpp>
#include <Builtins_generated.h>

namespace mz::vkss
{
template <typename T>
concept ResourceConcept = std::is_same_v<vk::Image, T> || std::is_same_v<vk::Buffer, T>;

template<typename T> struct ResourceTypeTo;
template<> struct ResourceTypeTo<vk::Image> {using CreateInfo = vk::ImageCreateInfo; };
template<> struct ResourceTypeTo<vk::Buffer> {using CreateInfo = vk::BufferCreateInfo; };

class ResourceManager
{
public:
	template <ResourceConcept T>
	vk::rc<T> Create(typename ResourceTypeTo<T>::CreateInfo const& info, std::string tag = "")
	{
		std::scoped_lock guard(Mutex);
		return GetPool<T>().Get(info, std::move(tag));
	}

	template <ResourceConcept T>
	bool Destroy(T* res)
	{
		std::scoped_lock guard(Mutex);
		auto impIt = ImportedResources.find(static_cast<vk::DeviceChild*>(res));
		if (impIt != ImportedResources.end())
		{
			ImportedResources.erase(impIt);
			return true;
		}
		return GetPool<T>().Release(res);
	}

	template <ResourceConcept T>
	bool Destroy(uint64_t handle)
	{
		return Destroy<T>(reinterpret_cast<T*>(handle));
	}

	template <ResourceConcept T>
	auto& GetPool()
	{
		if constexpr (std::is_same_v<vk::Image, T>)
			return ImagePool;
		if constexpr (std::is_same_v<vk::Buffer, T>)
			return BufferPool;
	}

	template <ResourceConcept T>
	void AddImported(vk::rc<T> res, std::string tag = "")
	{
		std::scoped_lock guard(Mutex);
		auto &info = ImportedResources[res.get()];
		info.Tag = std::move(tag);
		info.Resource = std::move(res);
	}
	
	template <ResourceConcept T>
	bool Exists(T* res)
	{
		std::scoped_lock guard(Mutex);
		if (ImportedResources.contains(reinterpret_cast<vk::DeviceChild*>(res)))
			return true;
		return GetPool<T>().IsUsed(res);
	}
	
	void Clear();
	vk::rc<vk::Image> GetColorTexture(glm::vec4 inColor);
	vk::rc<vk::Image> GetStockTexture(u32 width, u32 height, mz::fb::Format fmt, mz::fb::StockTexture stockType);
	vk::rc<vk::Image> GetStockTextureDirect(mz::fb::StockTexture stockType);
	void LoadStockTextures();

	// TODO: Port
	// void GetStatistics(editor::TEngineMetrics& metrics);
	// void GetResourceInfos(editor::TResources& resources);

protected:
	std::recursive_mutex Mutex; // TODO: To remove this mutex, use LockedSingleton for Scheduler or ensure this is called from single thread
	
	// In-process resources
	vkss::ImagePool ImagePool;
	vkss::BufferPool BufferPool;
	std::unordered_map<u32, vk::rc<vk::Image>>  Colors;
	std::unordered_map<u64, vk::rc<vk::Image>> ResizedStockTextures;
	std::unordered_map<mz::fb::StockTexture, vk::rc<vk::Image>> SrcStockTextures;

	// Imported
	struct ImportedResourceInfo
	{
		std::string Tag;
		vk::rc<vk::DeviceChild> Resource;
	};
	std::unordered_map<vk::DeviceChild*, ImportedResourceInfo> ImportedResources;
};

} // namespace mz::engine