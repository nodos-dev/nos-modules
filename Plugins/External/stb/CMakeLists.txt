cmake_minimum_required(VERSION 3.0)

project(stb C)

add_library(stb ${CMAKE_CURRENT_SOURCE_DIR}/stb.c)

target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
