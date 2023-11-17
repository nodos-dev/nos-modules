#pragma once

#include "mzUtil/Thread.h"
#include "mzVulkan/Common.h"
#include "mzVulkan/Semaphore.h"
#include "mzUtil/Singleton.h"

#include "MediaZ/Types.h"

namespace mz::vkss
{

#ifndef MZ_DISABLE_RENDER_THREAD
#define MZ_DISABLE_RENDER_THREAD 0
#endif

struct RenderEvent
{
	RenderEvent();
	RenderEvent(RenderEvent&& other);
	RenderEvent(const RenderEvent& other);
	vk::rc<vk::Semaphore> EventSemaphore = nullptr;

	void Wait();
};

struct RenderTask
{
	RenderTask() {};
	RenderTask(std::string name, std::function<void(vk::rc<mz::vk::CommandBuffer>)>&& task) : Name(std::move(name)), Task(std::move(task)) {};
	RenderTask(RenderTask&& other);
	RenderTask& operator=(RenderTask&& other);

	std::function<void(vk::rc<mz::vk::CommandBuffer>)> Task;
	std::string Name;
	
	RenderTask(const RenderTask& other) { assert(false); }
	RenderTask& operator=(const RenderTask& other) = delete;
};

#define CAT_(x, y, z) #x#y#z
#define CAT(...) CAT_(__VA_ARGS__)

#define ENQUEUE_RENDER_COMMAND(name, ...) Use<RenderThread>()->Enqueue(CAT(name,__FILE__,__LINE__), __VA_ARGS__)
#define ENQUEUE_RENDER_COMMAND_WITH_EVENT(name, ...) Use<RenderThread>()->EnqueueWithEvent(CAT(name,__FILE__,__LINE__), __VA_ARGS__)

class RenderThread : public ConsumerThread<RenderTask, 10000, MZ_DISABLE_RENDER_THREAD>, public Singleton<RenderThread>
{
public:
	using TaskType = Type;
	DECLARE_SINGLETON(RenderThread);

	void OnThreadStart() override;

	void Consume(const TaskType& item) override;

	void Flush();

	inline size_t NonThreadSafeQueueSize()
	{
		return Queue.NonThreadSafeSize();
	}

	inline void Enqueue(std::string name, std::function<void(vk::rc<mz::vk::CommandBuffer>)>&& task) 
	{
		ConsumerThread::Enqueue(RenderTask(std::move(name), std::move(task)));
	}
	RenderEvent EnqueueWithEvent(std::string name, std::function<void(vk::rc<mz::vk::CommandBuffer>)>&& item);

	void SetMarker(std::string marker);
	void ClearMarker();
	std::string Marker;

	void Stop() override;

	vk::rc<mz::vk::CommandPool> RTCommandPool;
	vk::rc<mz::vk::CommandBuffer> RTCurrentCmd;
};

}