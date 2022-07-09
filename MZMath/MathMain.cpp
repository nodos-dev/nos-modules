#include "mzMathConfig.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include <algorithm>
#include "Args.h"

using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

//template<typename T>
//void Add(void** inout, const char* metaData)
//{
//	Args params(inout, metaData);
//	params.Get<T>(3) = params.Get<T>(1) + params.Get<T>(2);
//}

extern "C"
{


void mzMath_API Add(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f32& X = params.Get<f32>("X");
	f32& Y = params.Get<f32>("Y");
	f32& Z = params.Get<f32>("Z");

	Z = X + Y;
}

void mzMath_API MakeVec4(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f32& X = params.Get<f32>(1);
	f32& Y = params.Get<f32>(2);
	f32& Z = params.Get<f32>(3);
	f32& W = params.Get<f32>(4);
	f32& V = params.Get<f32>(5);

	(&V)[0] = X;
	(&V)[1] = Y;
	(&V)[2] = Z;
	(&V)[3] = W;
}

void mzMath_API Sub(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f32& X = params.Get<f32>(1);
	f32& Y = params.Get<f32>(2);
	f32& Z = params.Get<f32>(3);

	Z = X - Y;
}

void mzMath_API SquareRoot(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);

	f32& X = params.Get<f32>("X");
	f32& Z = params.Get<f32>("Z");

	Z = sqrtf(X);
}

void mzMath_API Square(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f32& X = params.Get<f32>("X");
	f32& Z = params.Get<f32>("Z");

	Z = X * X;
}

}
