#pragma once
#include <MediaZ/PluginAPI.h>
#include <chrono>
class mzWebRTCStatsLogger
{
public:
	mzWebRTCStatsLogger(std::string name, int refreshRate = 100) : RefreshRate(refreshRate) 
	{
		Name_FPS = name + " FPS: ";
		Name_MAX_FPS = name + " MAX FPS: ";
		Name_MIN_FPS = name + " MIN FPS: ";
		startTime = std::chrono::high_resolution_clock::now();
	};

	void LogStats() {
		if (++FrameCount > RefreshRate)
		{
			// Clear stats for each 100 frame
			FrameCount = 0;
			MaxFPS = INT_MIN;
			MinFPS = INT_MAX;
		}
		auto now = std::chrono::high_resolution_clock::now();
		auto FPS = 1.0 / (std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count()) * 1000.0;
		MaxFPS = (FPS > MaxFPS && FPS < 500) ? (FPS) : (MaxFPS);
		MinFPS = (MinFPS > FPS && FPS > 5) ? (FPS) : (MinFPS);
		mzEngine.WatchLog(Name_FPS.c_str(), std::to_string(FPS).c_str());
		mzEngine.WatchLog(Name_MAX_FPS.c_str() , std::to_string(MaxFPS).c_str());
		mzEngine.WatchLog(Name_MIN_FPS.c_str(), std::to_string(MinFPS).c_str());
		startTime = now;
	}

private:
	std::string Name_FPS, Name_MAX_FPS, Name_MIN_FPS;
	int FrameCount = 0;
	int RefreshRate = 100;
	std::chrono::steady_clock::time_point startTime;
	int MinFPS = INT_MAX, MaxFPS = INT_MIN, FPS = 0;

};