#include "WindowNode.h"
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"

#include "nosUtil/Stopwatch.hpp"
namespace nos::test
{

void RegisterWindowNode(nosNodeFunctions* out)
{	
	NOS_BIND_NODE_CLASS(NOS_NAME_STATIC("nos.test.Window"), WindowNode, out);
}

WindowNode::WindowNode(const nos::fb::Node* node) : nos::NodeContext(node) 
{ 
	InitWindow(); 
}

WindowNode::~WindowNode() 
{ 
	Stop = true;
	if (Thread.joinable())
		Thread.join();
	//Cleanup semaphores
	for (int i = 0; i < FrameCount; i++)
	{
		nosVulkan->DestroySemaphore(&WaitSemaphore[i]);
		nosVulkan->DestroySemaphore(&SignalSemaphore[i]);
	}
}
void WindowNode::InitWindow()
{ 
	Thread = std::thread(&WindowNode::WindowThread, this); 
}

bool WindowNode::CreateSwapchain()
{
	nosSwapchainCreateInfo createInfo = {};
	createInfo.SurfaceHandle = Surface;
	createInfo.Extent = { 800, 600 };
	createInfo.PresentMode = NOS_PRESENT_MODE_FIFO;
	nosResult res = nosVulkan->CreateSwapchain(&createInfo, &Swapchain, &FrameCount);
	if (res != NOS_RESULT_SUCCESS)
		return false;
	nosSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.Type = NOS_SEMAPHORE_TYPE_BINARY;
	Images.resize(FrameCount);
	nosVulkan->GetSwapchainImages(Swapchain, Images.data());
	WaitSemaphore.resize(FrameCount);
	SignalSemaphore.resize(FrameCount);
	for (int i = 0; i < FrameCount; i++)
	{
		nosVulkan->CreateNosSemaphore(&semaphoreCreateInfo, &WaitSemaphore[i]);
		nosVulkan->CreateNosSemaphore(&semaphoreCreateInfo, &SignalSemaphore[i]);
	}
	return true;
}

void WindowNode::DestroySwapchain() 
{
	nosVulkan->DestroySwapchain(&Swapchain);
}

void WindowNode::DestroyWindowSurface() 
{
	nosVulkan->DestroyWindowSurface(&Surface);
}

void WindowNode::DestroyWindow() 
{
	glfwDestroyWindow(Window);
	glfwTerminate();
}

nosResult WindowNode::CopyTo(nosCopyInfo* cpy) 
{
	{
		std::unique_lock l2(PresentMutex);
		PresentCV.wait(l2, [&]() -> bool { return Presented; });
		Presented = false;
	}

	std::unique_lock lock(Mutex);
	nosCmd cmd;
	nosVulkan->Begin("Window Submit", &cmd);
	nosGPUEvent event;
	nosCmdEndParams endParams{.ForceSubmit = true, .OutGPUEventHandle = &event};
	nosVulkan->End(cmd, &endParams);
	util::Stopwatch sw;
	nosVulkan->WaitGpuEvent(&event, -1);
	nosEngine.LogI("Window: %s", sw.ElapsedString().c_str());
	{
		std::unique_lock l2(ReadyMutex);
		Ready = true;
		ReadyCV.notify_one();
	}
	Input = vkss::DeserializeTextureInfo(cpy->CopyToOptions.IncomingPinData->Data);
	return NOS_RESULT_SUCCESS;
}

void WindowNode::WindowThread() 
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	Window = glfwCreateWindow(800, 600, "NOSWindow", nullptr, nullptr);
	if (nosVulkan->CreateWindowSurface(glfwGetWin32Window(Window), &Surface) == NOS_RESULT_SUCCESS)
	{
		if (CreateSwapchain())
		{
		}
		else
		{
			DestroyWindowSurface();
			DestroyWindow();
			return;
		}
	}
	else
	{
		DestroyWindow();
		return;
	}
	
	nosScheduleNodeParams params = {};
	params.NodeId = NodeId;
	params.Reset = false;
	params.AddScheduleCount = 1;
	Ready = true;
	while (!glfwWindowShouldClose(Window) && !Stop)
	{
		uint32_t imageIndex;
		{
			std::unique_lock l2(ReadyMutex);
			ReadyCV.wait(l2, [&]() -> bool { return Ready; });
			Ready = false;
		}
		nosVulkan->SwapchainAcquireNextImage(Swapchain, -1, &imageIndex, WaitSemaphore[CurrentFrame]);
		nosCmd cmd;
		std::unique_lock lock(Mutex);
		{
			nosVulkan->Begin("Window", &cmd);
		
			if (Input.Info.Type == NOS_RESOURCE_TYPE_TEXTURE)
				nosVulkan->Copy(cmd, &Input, &Images[imageIndex], 0);
		
			nosVulkan->ImageStateToPresent(cmd, &Images[imageIndex]);
			nosVulkan->AddWaitSemaphoreToCmd(cmd, WaitSemaphore[CurrentFrame], 1);
			nosVulkan->AddSignalSemaphoreToCmd(cmd, SignalSemaphore[CurrentFrame], 1);
		
			nosCmdEndParams endParams{.ForceSubmit = true};
			nosVulkan->End(cmd, &endParams);
		}
		nosVulkan->SwapchainPresent(Swapchain, imageIndex, SignalSemaphore[CurrentFrame]);
		
		{
			std::unique_lock l2(PresentMutex);
			Presented = true;
			PresentCV.notify_one();
		}

		nosEngine.ScheduleNode(&params);
		glfwPollEvents();
		CurrentFrame = (CurrentFrame + 1) % FrameCount;
	}
	DestroySwapchain();
	DestroyWindowSurface();
	DestroyWindow();
}

}