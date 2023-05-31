// Copyright MediaZ AS. All Rights Reserved.

#include "Track.h"

MZ_INIT();

namespace mz
{

enum class TrackNode
{
	FreeD,
	Xync,
	SType,
	MoSys,
	UserTrack,
	CalculateNodalPoint,
	Count
};

void RegisterFreeDNode(mzNodeFunctions& functions);
void RegisterXyncNode(mzNodeFunctions& functions);
void RegisterStypeNode(mzNodeFunctions& functions);
void RegisterMoSysNode(mzNodeFunctions& functions);
void RegisterController(mzNodeFunctions& functions);
void RegisterCalculateNodalPoint(mzNodeFunctions& functions);

extern "C"
MZAPI_ATTR mzResult MZAPI_CALL mzExportNodeFunctions(size_t* outSize, mzNodeFunctions* outFunctions)
{
	*outSize = (size_t)TrackNode::Count;
	if (!outFunctions)
		return MZ_RESULT_SUCCESS;
	RegisterFreeDNode(outFunctions[0]);
	RegisterXyncNode(outFunctions[1]);
	RegisterStypeNode(outFunctions[2]);
	RegisterMoSysNode(outFunctions[3]);
	RegisterController(outFunctions[4]);
	RegisterCalculateNodalPoint(outFunctions[5]);
	return MZ_RESULT_SUCCESS;
}

} // namespace mz