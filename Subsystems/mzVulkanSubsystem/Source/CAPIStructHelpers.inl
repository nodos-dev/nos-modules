//Structs for deep copying C API structs & scoped memory management

template<class T> 
struct OwnedInfoTraitsBindings
{
    inline static constexpr auto Data  = &T::Bindings;
    inline static constexpr auto Count = &T::BindingCount;
    inline static constexpr bool NeedsOwning = true;
    using Type = mzShaderBinding;
};

template<class T> 
struct OwnedInfoTraits
{
    inline static constexpr bool NeedsOwning = false;
};

template<> struct OwnedInfoTraits<mzDrawCall> : OwnedInfoTraitsBindings<mzDrawCall> {};
template<> struct OwnedInfoTraits<mzRunPassParams> : OwnedInfoTraitsBindings<mzRunPassParams> {};
template<> struct OwnedInfoTraits<mzRunComputePassParams> : OwnedInfoTraitsBindings<mzRunComputePassParams> {};

template<> 
struct OwnedInfoTraits<mzShaderBinding>
{
    inline static constexpr bool NeedsOwning = true;
    inline static constexpr auto Data  = &mzShaderBinding::Data;
    inline static constexpr auto Count = &mzShaderBinding::Size;
    using Type = u8;
};

template<> 
struct OwnedInfoTraits<mzRunPass2Params>
{
    inline static constexpr bool NeedsOwning = true;
    inline static constexpr auto Data  = &mzRunPass2Params::DrawCalls;
    inline static constexpr auto Count = &mzRunPass2Params::DrawCallCount;
    using Type = mzDrawCall;
};

template<class T> 
struct OwnedInfoVector
{
    using traits = OwnedInfoTraits<T>;
    using inner = typename traits::Type;
    using data_type = std::conditional_t<OwnedInfoTraits<inner>::NeedsOwning, std::vector<OwnedInfoVector<inner>>, mz::Buffer>;

    OwnedInfoVector() = default;
    
	OwnedInfoVector(const T* infos, size_t count) requires(OwnedInfoTraits<typename OwnedInfoTraits<T>::Type>::NeedsOwning)  :
        Infos(infos, infos + count),
        Data{}
	{
        Data.reserve(count);
        for(auto& info : Infos)
        {
            Data.emplace_back(info.*traits::Data, info.*traits::Count);
            info.*traits::Data = Data.back().Infos.data();
        }
	}
    
	OwnedInfoVector(const T* infos, size_t count) requires(!OwnedInfoTraits<typename OwnedInfoTraits<T>::Type>::NeedsOwning) :
        Infos(infos, infos + count),
        Data(std::accumulate(infos, infos + count, 0ull, [](size_t l, auto& r) { return l + r.*OwnedInfoTraits<T>::Count; }))
	{
        u8* dat = Data.data();

        for(auto& b : Infos)
        {
            memcpy(dat, b.*OwnedInfoTraits<T>::Data, b.*OwnedInfoTraits<T>::Count);
            b.*OwnedInfoTraits<T>::Data = (typename OwnedInfoTraits<T>::Type*)dat;
            dat += b.*OwnedInfoTraits<T>::Count;
        }
	}

	OwnedInfoVector(OwnedInfoVector&& other) = default;
    OwnedInfoVector& operator=(OwnedInfoVector&&) = default;
	OwnedInfoVector(const OwnedInfoVector& other) 
        : OwnedInfoVector(other.Infos.data(), other.Infos.size())
	{
	}
    
	std::vector<T> Infos;
	data_type Data;
};


template<class T>
struct OwnedInfoStruct : T, OwnedInfoVector<typename OwnedInfoTraits<T>::Type>
{
    OwnedInfoStruct(const T& from) : T(from), OwnedInfoVector<typename OwnedInfoTraits<T>::Type>(from.*OwnedInfoTraits<T>::Data, from.*OwnedInfoTraits<T>::Count)
    {
        static_cast<T*>(this)->*OwnedInfoTraits<T>::Data = static_cast<OwnedInfoVector<typename OwnedInfoTraits<T>::Type>*>(this)->Infos.data();
    }

    OwnedInfoStruct(OwnedInfoStruct&&) = default;
    OwnedInfoStruct& operator=(OwnedInfoStruct&&) = default;
    
    OwnedInfoStruct(const OwnedInfoStruct& from) : OwnedInfoStruct(static_cast<const T&>(from)) {}
    OwnedInfoStruct& operator=(const OwnedInfoStruct& from)  
    {
        static_cast<T*>(this)->*OwnedInfoTraits<T>::Data = static_cast<OwnedInfoVector<typename OwnedInfoTraits<T>::Type>*>(this)->Infos.data();
    }
};

using OwnedRunPassParams = OwnedInfoStruct <mzRunPassParams>;
using OwnedRunPass2Params = OwnedInfoStruct<mzRunPass2Params>;
using OwnedRunComputePassParams = OwnedInfoStruct<mzRunComputePassParams>;

