/*
 * Copyright MediaZ Teknoloji A.S. All Rights Reserved.
 */

#pragma once
#include <Nodos/PluginAPI.h>
#include <chrono>
class nosWebRTCStatsLogger
{
public:
	nosWebRTCStatsLogger(std::string name, int refreshRate = 100) : RefreshRate(refreshRate) 
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
			MaxFPS = -INFINITY;
			MinFPS = INFINITY;
		}
		auto now = std::chrono::high_resolution_clock::now();
		auto FPS = 1.0 / (std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count()) * 1000.0;
		MaxFPS = (FPS > MaxFPS) ? (FPS) : (MaxFPS);
		MinFPS = (MinFPS > FPS) ? (FPS) : (MinFPS);       
		nosEngine.WatchLog(Name_FPS.c_str(), std::to_string(FPS).c_str());
		nosEngine.WatchLog(Name_MAX_FPS.c_str() , std::to_string(MaxFPS).c_str());
		nosEngine.WatchLog(Name_MIN_FPS.c_str(), std::to_string(MinFPS).c_str());
		startTime = now;
	}

	float GetMaxFPS() const {
		return MaxFPS;
	}

	float GetMinFPS() const {
		return MinFPS;
	}

private:
	std::string Name_FPS, Name_MAX_FPS, Name_MIN_FPS;
	int FrameCount = 0;
	int RefreshRate = 100;
	std::chrono::steady_clock::time_point startTime;
	float MinFPS = 9999, MaxFPS = -9999, FPS = 0;

};

struct WebRTCRing
{
    struct
    {
        std::deque<int> Pool;
        std::mutex Mutex;
        std::condition_variable CV;
    } Write, Read;

    std::string RingName;
    u32 Size = 0;
    nosVec2u Extent;
    std::atomic_bool Exit = false;
    std::atomic_bool ResetFrameCount = true;
   
    WebRTCRing(u32 ringSize, std::string ringName) : RingName(std::move(ringName))
    {
        Resize(ringSize);
    }

    ~WebRTCRing()
    {
        Stop();
    }
    
    void Resize(u32 size)
    {
        Write.Pool = {};
        Read.Pool = {};
        for (u32 i = 0; i < size; ++i)
        {
            Write.Pool.push_back(i);
        }
        Size = size;
    }

    void Stop()
    {
        {
            std::unique_lock l1(Write.Mutex);
            std::unique_lock l2(Read.Mutex);
            Exit = true;
        }
        Write.CV.notify_all();
        Read.CV.notify_all();
    }

    bool IsFull()
    {
        std::unique_lock lock(Read.Mutex);
        return Read.Pool.size() == Size;
    }

    bool HasEmptySlots()
    {
        return EmptyFrames() != 0;
    }

    u32 EmptyFrames()
    {
        std::unique_lock lock(Write.Mutex);
        return Write.Pool.size();
    }

    bool IsEmpty()
    {
        std::unique_lock lock(Read.Mutex);
        return Read.Pool.empty();
    }

    u32 ReadyFrames()
    {
        std::unique_lock lock(Read.Mutex);
        return Read.Pool.size();
    }

    u32 TotalFrameCount()
    {
        std::unique_lock lock(Write.Mutex);
        return Size - Write.Pool.size();
    }

    std::optional<int> BeginPush()
    {
        std::unique_lock lock(Write.Mutex);
        while (Write.Pool.empty() && !Exit)
            Write.CV.wait(lock);
        if (Exit)
            return std::nullopt;
        int res = Write.Pool.front();
        Write.Pool.pop_front();
        return res;
    }

    void EndPush(int res)
    {
        {
            std::unique_lock lock(Read.Mutex);
            Read.Pool.push_back(res);
            assert(Read.Pool.size() <= Size);
        }
        Read.CV.notify_one();
    }

    void CancelPush(int res)
    {
        {
            std::unique_lock lock(Write.Mutex);
            Write.Pool.push_front(res);
            assert(Write.Pool.size() <= Size);
        }
        Write.CV.notify_one();
    }
    void CancelPop(int res)
    {
        {
            std::unique_lock lock(Read.Mutex);
            Read.Pool.push_front(res);
            assert(Read.Pool.size() <= Size);
        }
        Read.CV.notify_one();
    }

    std::optional<int> BeginPop(uint64_t timeoutMilliseconds)
    {
        std::unique_lock lock(Read.Mutex);
        if (!Read.CV.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds), [this]() {return !Read.Pool.empty() || Exit; }))
            return std::nullopt;
        if (Exit)
            return std::nullopt;
        auto res = Read.Pool.front();
        Read.Pool.pop_front();
        return res;
    }

    void EndPop(int res)
    {
        {
            std::unique_lock lock(Write.Mutex);
            Write.Pool.push_back(res);
            assert(Write.Pool.size() <= Size);
        }
        Write.CV.notify_one();
    }

    bool CanPop(u32 spare = 0)
    {
        std::unique_lock lock(Read.Mutex);
        if (Read.Pool.size() > spare)
        {
            // TODO: Under current arch, schedule requests are sent for the node instead of pin, so this code shouldn't be needed, but check.
            // auto newFrameNumber = Read.Pool.front()->FrameNumber.load();
            // bool result = ResetFrameCount || !frameNumber || newFrameNumber > frameNumber;
            // frameNumber = newFrameNumber;
            // ResetFrameCount = false;
            return true;
        }

        return false;
    }

    bool CanPush()
    {
        std::unique_lock lock(Write.Mutex);
        return !Write.Pool.empty();
    }

    std::optional<int> TryPush()
    {
        if (CanPush())
            return BeginPush();
        return std::nullopt;
    }

    std::optional<int> TryPush(const std::chrono::milliseconds timeout)
    {
        {
            std::unique_lock lock(Write.Mutex);
            if (Write.Pool.empty())
                Write.CV.wait_for(lock, timeout, [&] { return CanPush(); });
        }
        return TryPush();
    }

    std::optional<int> TryPop(u32 spare = 0)
    {
        if (CanPop(spare))
            return BeginPop(20);
        return std::nullopt;
    }

    void Reset(bool fill)
    {
        auto& from = fill ? Write : Read;
        auto& to = fill ? Read : Write;
        std::unique_lock l1(Write.Mutex);
        std::unique_lock l2(Read.Mutex);
        while (!from.Pool.empty())
        {
            int slot = from.Pool.front();
            from.Pool.pop_front();
            to.Pool.push_back(slot);
        }
    }

    void LogRing() {
        nosEngine.WatchLog((RingName + " ready frames").c_str(), std::to_string(Read.Pool.size()).c_str());
        nosEngine.WatchLog((RingName + " empty frames").c_str(), std::to_string(Write.Pool.size()).c_str());
    }
};