add_library(mzMediaSource SHARED MediaSource.cpp)

target_link_libraries(mzMediaSource PRIVATE mzVulkan)

compile_protos(mzMediaSource)