if (NOT WITH_NODOS AND NOT TARGET stb)
    add_subdirectory(stb "${CMAKE_CURRENT_BINARY_DIR}/stb")
endif()