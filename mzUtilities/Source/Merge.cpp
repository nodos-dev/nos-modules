// Copyright MediaZ AS. All Rights Reserved.

#include "Merge.hpp"
#include "Merge.frag.spv.dat"

#include <random>

namespace mz
{
    std::seed_seq Seed()
{
    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size>{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    return std::seed_seq(std::begin(seed_data), std::end(seed_data));
}

}

namespace mz::filters
{

struct MergeContext
{
    MzResourceShareInfo dummyTexture;

    uint32_t inputCount;

    ~MergeContext()
    {
        mzEngine.Destroy(&dummyTexture);
    }

    void Run(const MzNodeExecuteArgs* pins)
    {
        std::vector<MzResourceShareInfo> textures;
        int Count = 0;
        for(size_t i{}; i < pins->PinCount; ++i)
        {
            if(pins->PinNames[i] == "Output")
                continue;
            textures.push_back(ValAsTex(pins->PinValues[i].Data));
        }

        for(size_t i{}; i < pins->PinCount; ++i)
            if((std::string(pins->PinNames[i])).find("Texture_") != std::string::npos)
                ++Count;
        
        for(size_t i = Count; i <16; ++i)
        {

        }
    }
};

}