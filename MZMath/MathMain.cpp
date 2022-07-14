#include "mzMathConfig.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include <algorithm>
#include "Args.h"
#include "Builtins.pb.h"

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
	
	 auto& X = params.Get<mz::proto::f32>("X");
	 auto& Y = params.Get<mz::proto::f32>("Y");
	 auto& Z = params.Get<mz::proto::f32>("Z");

	Z.set_val(X.val() + Y.val());
}


void mzMath_API Addf64(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f64& X = params.Get<f64>("X");
	f64& Y = params.Get<f64>("Y");
	f64& Z = params.Get<f64>("Z");

	Z = X + Y;
}

void mzMath_API MakeVec4(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f32& X = params.Get<f32>("X");
	f32& Y = params.Get<f32>("Y");
	f32& Z = params.Get<f32>("Z");
	f32& W = params.Get<f32>("W");
	f32& V = params.Get<f32>("V");

	(&V)[0] = X;
	(&V)[1] = Y;
	(&V)[2] = Z;
	(&V)[3] = W;
}

void mzMath_API MakeVec3(void** inout, const char* metaData)
{
	mz::Args params(inout, metaData);
	
	f64& X = params.Get<f64>("X");
	f64& Y = params.Get<f64>("Y");
	f64& Z = params.Get<f64>("Z");
	f64& V = params.Get<f64>("V");

	(&V)[0] = X;
	(&V)[1] = Y;
	(&V)[2] = Z;
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
