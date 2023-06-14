#include <MediaZ/Helpers.hpp>

namespace mz::utilities
{

MZ_REGISTER_NAME_SPACED(Indexer, "mz.utilities.Indexer")
MZ_REGISTER_NAME(Output);

void RegisterIndexer(mzNodeFunctions* fn)
{
	fn->TypeName = MZN_Indexer;
    
	fn->OnNodeCreated = [](const mzFbNode* node, void** outCtxPtr) 
    {
	};

	fn->OnNodeDeleted = [](void* ctx, mzUUID nodeId) 
    {
	};

	fn->ExecuteNode = [](void* ctx, const mzNodeExecuteArgs* args) -> mzResult 
    {
        return MZ_RESULT_SUCCESS;
	};
}

} // namespace mz