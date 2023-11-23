#include "Globals.h"
#include "Resources/ResourceManager.h"

namespace mz::vkss
{
mz::vk::Device* GVkDevice;
vk::rc<mz::vk::Context> GVkCtx;
std::unique_ptr<ResourceManager> GResources{};
}