if (NOT WITH_NODOS AND NOT TARGET stb)
    project(stb C)

    add_library(stb ${CMAKE_CURRENT_SOURCE_DIR}/stb/stb_vorbis.c)

    target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()