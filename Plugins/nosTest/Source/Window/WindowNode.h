#pragma once

#include "Nodos/PluginHelpers.hpp"
#include "nosVulkanSubsystem/Helpers.hpp"

#include "GLFW/glfw3.h"

namespace nos::test
{
void RegisterWindowNode(nosNodeFunctions* out);
class WindowNode : public nos::NodeContext
{
public:
	WindowNode(const nos::fb::Node* node);
	~WindowNode();
	void InitWindow();
	bool CreateSwapchain();
	void DestroySwapchain();
	void DestroyWindowSurface();
	void DestroyWindow();

	nosResult CopyTo(nosCopyInfo* cpy) override;

private:
	void WindowThread();

	GLFWwindow* Window = nullptr;
	std::vector<nosSemaphore> WaitSemaphore{};
	std::vector<nosSemaphore> SignalSemaphore{};
	std::vector<nosResourceShareInfo> Images{};
	uint32_t FrameCount = 0;
	uint32_t CurrentFrame = 0;
	nosSurfaceHandle Surface{};
	nosSwapchainHandle Swapchain{};
	std::thread Thread{};
	std::atomic_bool Stop = false;
	nosResourceShareInfo Input{};
	std::mutex Mutex{};

	std::atomic_bool Presented = false;
	std::condition_variable PresentCV{};
	std::mutex PresentMutex{};

	std::atomic_bool Ready = false;
	std::condition_variable ReadyCV{};
	std::mutex ReadyMutex{};
};
}