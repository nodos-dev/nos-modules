// Copyright MediaZ AS. All Rights Reserved.

#include "AppService_generated.h"
#include "BasicMain.h"
#include "mzFlatBuffersCommon.h"

#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>
#include <mutex>

namespace mz
{


struct DelayContext: PinMapping
{
    u32 Delay   = 0;

    std::queue<mz::Buffer> Ring;
    
    virtual void Push(mz::Buffer const& buf)
    {
        Ring.push(mz::Buffer(buf));
    }
    
    virtual void Pop(mz::Buffer* outBuf)
    {
        while(Ring.size() > Delay + 1)
        {
            Ring.pop();
        }
        if(Ring.empty() || Ring.size() < Delay)
        {
            return;
        }
        *outBuf = std::move(Ring.front());
        Ring.pop();
    }

    virtual void Repeat()
    {
        Push(Ring.back());
    }
    
    virtual ~DelayContext() = default;
};

struct JobQueue
{
    std::mutex Mutex;
    std::condition_variable CV;
    std::vector<std::function<void()>> Jobs;
    std::thread Thread;
    std::atomic_bool Exit = true;

    JobQueue() 
    {
        Start();
    }

    void Start()
    {
        if(!Exit) return;
        Exit = false;
        Thread = std::thread([this] {
            while(!Exit)
            {
                std::vector<std::function<void()>> jobs;
                {
                    std::unique_lock lock(Mutex);
                    while(!Exit && Jobs.empty())
                    {
                        CV.wait(lock);
                    }
                    if(Exit) break;
                    jobs = std::move(Jobs);
                }

                for(auto& job : jobs)
                {
                    job();
                }
            }
        });
    }

    void Enqueue(std::function<void()>&& job)
    {
        {
            std::unique_lock lock(Mutex);
            Jobs.emplace_back(std::move(job));
        }
        CV.notify_all();
    }

    void Stop() 
    {
        if(Exit) return;
        {
            std::unique_lock lock(Mutex);
            Exit = true;
        }
        Thread.join();
    }

    ~JobQueue() { Stop(); }
};


// TODO: DelayResource with policy based handling of mz::fb::Buffer and mz::fb::TTexture
struct DelayTexture : DelayContext
{
    mz::fb::TTexture Base;
    //JobQueue Queue;
    //
    std::queue<mz::fb::TTexture> FreeList;

    mz::fb::TTexture GetTexture()
    {
        mz::fb::TTexture tmp;

        if (Ring.size() > Delay)
        {
            tmp = Ring.front().As<mz::fb::TTexture>();
            Ring.pop();
        }
        else if (!FreeList.empty())
        {
            tmp = FreeList.front();
            FreeList.pop();
        }
        else
        {
            tmp = Base;
            mzEngine.Create(tmp);
        }
        
        return tmp;
    }

    void Push(mz::Buffer const& buf) override
    {
        auto tmp = GetTexture();
        Ring.push(Buffer::From(tmp));
        auto src = TableRootToNativeTable(buf.As<mz::fb::Texture>());
        auto dst = tmp;
        //Queue.Enqueue([this, src = src, dst = dst] {
            app::TBlitTexture blit;
            blit.src = std::make_unique<mz::fb::TTexture>(src);
            blit.dst = std::make_unique<mz::fb::TTexture>(dst);
            mzEngine.MakeAPICalls(true, blit);
        //});
    }
    
    void Pop(mz::Buffer* buf) override
    {
        while(Ring.size() > Delay + 1)
        {
            FreeList.push(Ring.front().As<mz::fb::TTexture>());
            Ring.pop();
        }
        if (Ring.empty() || Ring.size() < Delay)
            return;
        FreeList.push(buf->As<mz::fb::TTexture>());
        *buf = std::move(Ring.front());
        Ring.pop();
    }
};

void RegisterDelay(NodeActionsMap& functions, std::set<flatbuffers::Type const*> const& types)
{
    mz::mzEngine = mzEngine;

    for(auto const& type : types)
    {
        auto className = GetTypeName(type);
        auto& action = functions["delay." + className];
        action.NodeCreated = [type](fb::Node const& nodeDef, mz::Args& args, void** ctx)
        { 
            auto name = GetTypeName(type);
            DelayContext* delay = nullptr;
            if("mz.fb.Texture" == name)
                delay = new DelayTexture{};
            else 
                delay = new DelayContext{};
            *ctx = delay;
            if(auto delayPin = delay->Load(nodeDef)["Delay"])
            {
                if(flatbuffers::IsFieldPresent(delayPin, mz::fb::Pin::VT_DATA))
                {
                    delay->Delay = *(u32*)delayPin->data()->data();
                }
            }
        };

        action.NodeRemoved = [](void* ctx, mz::fb::UUID const& id)
        {
            delete static_cast<DelayContext*>(ctx);
        };
        
        action.PinValueChanged = [](void* ctx, mz::fb::UUID const& id, mz::Buffer* value)
        {
            DelayContext* delayRing = static_cast<DelayContext*>(ctx);
            auto name = delayRing->GetPinName(id);
            if ("Input" == name)
            {
                delayRing->Push(*value);
                return;
            }
            if ("Delay" == name)
            {
                delayRing->Delay = *reinterpret_cast<u32*>(value->data());
                return;
            }
        };
        
        action.EntryPoint = [](mz::Args& args, void* ctx)
        {
            auto* delayRing = static_cast<DelayContext*>(ctx);
            delayRing->Pop(args.GetBuffer("Output"));
            return true; 
        };
    }
}

}