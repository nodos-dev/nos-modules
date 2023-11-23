#include "RenderThread.h"

#include "Globals.h"

// Vulkan base
#include <mzVulkan/Device.h>
#include <mzVulkan/Command.h>

// External
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <processthreadsapi.h>

// std
#include <codecvt>

namespace mz::vkss
{
using namespace mz::vk;

IMPLEMENT_SINGLETON(RenderThread);

// #define VERBOSE_RENDER_THREAD

void SetThreadName(HANDLE handle, std::string const& threadName)
{
	const uint32_t MS_VC_EXCEPTION = 0x406D1388;

	struct THREADNAME_INFO
	{
		uint32_t dwType;	 // Must be 0x1000.
		LPCSTR szName;		 // Pointer to name (in user addr space).
		uint32_t dwThreadID; // Thread ID (-1=caller thread).
		uint32_t dwFlags;	 // Reserved for future use, must be zero.
	};

	THREADNAME_INFO ThreadNameInfo;
	ThreadNameInfo.dwType = 0x1000;
	ThreadNameInfo.szName = threadName.c_str();
	ThreadNameInfo.dwThreadID = ::GetThreadId(handle);
	ThreadNameInfo.dwFlags = 0;

	__try
	{
		RaiseException(MS_VC_EXCEPTION, 0, sizeof(ThreadNameInfo) / sizeof(ULONG_PTR), (ULONG_PTR*)&ThreadNameInfo);
	}
	__except (EXCEPTION_EXECUTE_HANDLER) __pragma(warning(suppress : 6322))
	{
	}
}

void RenderThread::OnThreadStart()
{

	std::string threadName = "=====> [MZ Render Thread]";

	HANDLE handle = ::GetCurrentThread();
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	auto threadNameW = converter.from_bytes(threadName);
	::SetThreadDescription((HANDLE)handle, threadNameW.c_str());
	SetThreadName((HANDLE)handle, threadName);

	//util::SetCurrentThreadName("MZ Render Thread");
	RTCommandPool = GVkDevice->GetPool();
	RTCurrentCmd = RTCommandPool->BeginCmd();
}

void RenderThread::Consume(const TaskType& task)
{
#ifdef VERBOSE_RENDER_THREAD 
	if(task.Name == std::string("SetMarker"))
		ll() << Marker << "->" << task.Name;
#endif
	task.Task(RTCurrentCmd);
	if (RTCurrentCmd->State != CommandBuffer::State::Recording)
		RTCurrentCmd = RTCommandPool->BeginCmd();
}

void RenderThread::Flush()
{
	ENQUEUE_RENDER_COMMAND_WITH_EVENT("Flush", [](vk::rc<vk::CommandBuffer> cmd) {
		cmd->Submit();
	}).Wait();
}

RenderEvent RenderThread::EnqueueWithEvent(std::string name, std::function<void(vk::rc<mz::vk::CommandBuffer>)>&& item)
{
	RenderEvent event;
	Enqueue(name, [event = event, item = std::move(item)](vk::rc<vk::CommandBuffer> cmd){
		cmd->SignalGroup[event.EventSemaphore->Handle] = 1;
		cmd->AddDependency(event.EventSemaphore);
		item(cmd);
	});
	return event;
}

void RenderThread::SetMarker(std::string marker)
{
#ifdef VERBOSE_RENDER_THREAD 
	ENQUEUE_RENDER_COMMAND(SetMarker, [this, marker = std::move(marker)](vk::rc<vk::CommandBuffer> cmd) {
		Marker = marker;
	});
#endif
}

void RenderThread::ClearMarker()
{
	SetMarker("Scheduler");
}

void RenderThread::Stop()
{
	Flush();
	RTCommandPool.reset();
	RTCurrentCmd.reset();
	ConsumerThread::Stop();
}

RenderEvent::RenderEvent() { EventSemaphore = vk::Semaphore::New(GVkDevice); }

RenderEvent::RenderEvent(RenderEvent&& other)
{
	EventSemaphore.swap(other.EventSemaphore);
}

RenderEvent::RenderEvent(const RenderEvent& other)
{
	EventSemaphore = other.EventSemaphore;
}

void RenderEvent::Wait()
{
	if (EventSemaphore)
		EventSemaphore->Wait(1);
}

RenderTask::RenderTask(RenderTask&& other) : Task(std::move(other.Task)), Name(std::move(other.Name)) {}

RenderTask& RenderTask::operator=(RenderTask&& other)
{
	Task = std::move(other.Task);
	Name = std::move(other.Name);
	return *this;
}

}

