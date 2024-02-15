#ifndef AI_COMMON_MACROS_H_INCLUDED
#define AI_COMMON_MACROS_H_INCLUDED
#include "Nodos/SubsystemAPI.h"

#define CHECK_NOS_RESULT(nosRes) \
	do { \
		nosResult __MACRO__RESULT__= nosRes; \
		if (__MACRO__RESULT__ != NOS_RESULT_SUCCESS) { \
			nosEngine.LogE("Failed from %s %d with error %S.",__FILE__, __LINE__,GetNosResultString(__MACRO__RESULT__)); \
			return NOS_RESULT_FAILED; \
		} \
	} while (0); \

#define CHECK_PATH(path) \
	do{ \
		if(!std::filesystem::exists(path)){\
			nosEngine.LogE("%s %s : Path %s not exists", __FILE__, __LINE__, path.string().c_str());\
		}\
	} while (0);\

#define CHECK_FILE_EXTENSION(filePath, desiredExtension)\
	do{ \
		std::filesystem::path __MACRO__PATH__(filePath);\
		if (__MACRO__PATH__.extension().string().compare(desiredExtension) != 0) {\
			nosEngine.LogE("Given file %s is not in the expected %s format", __MACRO__PATH__.string().c_str(), __MACRO__PATH__.extension().string().c_str());\
			return NOS_RESULT_FAILED;\
		}\
	} while (0);\

#define CHECK_POINTER(ptr)\
	do {\
		if (ptr == nullptr) {\
			return NOS_RESULT_FAILED;\
		}\
	} while (0);\

#define CHECK_INDEX_BOUNDS(index, Size)\
	do {\
		if (index >= Size) {\
			nosEngine.LogE("From %s %s : Index [%d] out of bounds [%d]",__FILE__,__LINE__,index, Size);\
			return NOS_RESULT_FAILED;\
		}\
	} while (0);\

#define CHECK_MODEL_FORMAT(model, desiredFormat)\
	do {\
		if (model->Format == desiredFormat) {\
			return NOS_RESULT_FAILED;\
		}\
	} while (0);\

#endif //AI_COMMON_MACROS_H_INCLUDED