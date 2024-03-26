# Copyright MediaZ Teknoloji A.S. All Rights Reserved.

cmake_minimum_required(VERSION 3.24.2)
project("Subsystems")

include(${SCRIPTS_DIR}/CMake/Subsystems.cmake)

# Update subsystems.json if you add dependencies, so that CD can create releases.

add_subdirectory(nosVulkanSubsystem) # Subsystem prefix is only for distinguishing between nosVulkan repo and this.
add_subdirectory(nosTestSubsystem)
add_subdirectory(nosPython)
