#pragma once


namespace mz
{

template<typename T>
struct TRing
{
    T Sample;
    
    struct Resource
    {
        MzResourceShareInfo Res;

        Resource(T r) : Res{}
        {
            if constexpr (std::is_same_v<T, MzBufferInfo>)
            {
                Res.info.type = MZ_RESOURCE_TYPE_BUFFER;
                Res.info.buffer = r;
            }
            else
            {
                Res.info.type = MZ_RESOURCE_TYPE_TEXTURE;
                Res.info.texture = r;
            }
            mzEngine.Create(&Res);
        }
        
        ~Resource()
        {
            mzEngine.Destroy(&Res);
        }

        std::atomic_uint32_t FrameNumber;
        std::atomic_bool written = false;
        std::atomic_bool read = false;
    };

    void Resizex(u32 size)
    {
        Write.Pool.clear();
        Read.Pool.clear();
        Glob.clear();
        for (u32 i = 0; i < size; ++i)
            Glob.push_back(MakeShared<Resource>(Sample));

        std::transform(Glob.begin(), Glob.end(), std::back_inserter(Write.Pool), [](auto rc) { return rc.get(); });
        Size = size;
    }
    
    TRing(MzVec2u extent, u32 Size) 
        requires(std::is_same_v<T, MzBufferInfo>)
        : Extent(extent), Sample()
    {
        Sample.size = Extent.x * Extent.y * 4;
        Resizex(Size);
    }
    
    TRing(MzVec2u extent, u32 Size, MzFormat format = MZ_FORMAT_R16G16B16A16_UNORM)
        requires(std::is_same_v<T, MzTextureInfo>)
        : Extent(extent), Sample()
    {
        Sample.width = Extent.x;
        Sample.height = Extent.y;
        Sample.format = format;
        Resizex(Size);
    }

    struct
    {
        std::vector<Resource *> Pool;
        std::mutex Mutex;
        std::condition_variable CV;
    } Write, Read;

    std::vector<rc<Resource>> Glob;

    u32 Size = 0;
    MzVec2u Extent;
    std::atomic_bool Exit = false;
    std::atomic_bool ResetFrameCount = true;

    ~TRing()
    {
        Stop();
        Glob.clear();
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
        std::unique_lock lock(Write.Mutex);
        return Write.Pool.empty();
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

    Resource *BeginPush()
    {
        std::unique_lock lock(Write.Mutex);
        while (Write.Pool.empty() && !Exit)
        {
            Write.CV.wait(lock);
        }
        if (Exit)
            return 0;
        Resource *res = Write.Pool.back();
        Write.Pool.pop_back();
        assert(!res->written || !res->read);
        res->written = true;
        return res;
    }

    void EndPush(Resource *res)
    {
        {
            std::unique_lock lock(Read.Mutex);
            assert(res->written || !res->read);
            res->written = false;
            Read.Pool.push_back(res);
        }
        Read.CV.notify_one();
    }

    Resource *BeginPop()
    {
        std::unique_lock lock(Read.Mutex);
        while (Read.Pool.empty() && !Exit)
        {
            Read.CV.wait(lock);
        }
        if (Exit)
            return 0;
        auto res = Read.Pool.front();
        Read.Pool.erase(Read.Pool.begin());
        assert(!res->written || !res->read);
        res->read = true;
        return res;
    }

    void EndPop(Resource *res)
    {
        {
            std::unique_lock lock(Write.Mutex);
            assert(!res->written || res->read);
            res->read = false;
            res->FrameNumber = 0;
            Write.Pool.push_back(res);
        }
        Write.CV.notify_one();
    }

    bool CanPop(u32& frameNumber, u32 spare = 0)
    {
        std::unique_lock lock(Read.Mutex);
        if (Read.Pool.size() > spare)
        {
            auto newFrameNumber = Read.Pool.front()->FrameNumber.load();
            bool result = ResetFrameCount || !frameNumber || newFrameNumber > frameNumber;
            frameNumber = newFrameNumber;
            ResetFrameCount = false;
            return result;
        }

        return false;
    }

    bool CanPush()
    {
        std::unique_lock lock(Write.Mutex);
        return !Write.Pool.empty();
    }

    Resource *TryPush()
    {
        if (!IsFull())
        {
            return BeginPush();
        }
        return 0;
    }

    Resource *TryPop(u32& frameNumber, u32 spare = 0)
    {
        if (CanPop(frameNumber, spare))
            return BeginPop();
        return 0;
    }
};

typedef TRing<MzBufferInfo> CPURing;
typedef TRing<MzTextureInfo> GPURing;

} // namespace mz