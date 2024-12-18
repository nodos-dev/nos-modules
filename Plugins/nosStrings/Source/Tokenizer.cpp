// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

namespace nos::strings
{
struct TokenizerNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		auto pin = GetPinValues(params);
		auto text = GetPinValue<const char>(pin, NOS_NAME("Text"));
		auto delimiter = GetPinValue<const char>(pin, NOS_NAME("Delimiter"));
		auto tokensPinId = *GetPinId(NOS_NAME("Tokens"));
		
		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream(text);
		while (std::getline(tokenStream, token, delimiter[0]))
		{
			tokens.push_back(token);
		}

		flatbuffers::FlatBufferBuilder fbb;
		std::vector<flatbuffers::Offset<flatbuffers::String>> elements;
		for (int i = 0; i < tokens.size(); ++i)
			elements.push_back(fbb.CreateString(tokens[i].c_str()));
		fbb.Finish(fbb.CreateVector(elements));
		nos::Buffer tokensBuf = fbb.Release();
		auto root = flatbuffers::GetRoot<uint8_t>(tokensBuf.Data()); // TODO: Move array creation code to API.
		nosEngine.SetPinValue(tokensPinId, { (void*)root, tokensBuf.Size() });
		return NOS_RESULT_SUCCESS;
	}
};


nosResult RegisterTokenizer(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Tokenizer"), TokenizerNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos