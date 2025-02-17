// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

// stl
#include <regex>

namespace nos::strings
{
struct RegexNode : NodeContext
{
	using NodeContext::NodeContext;

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nos::NodeExecuteParams pins(params);
		
		const char* expression = pins.GetPinData<const char>(NOS_NAME("RegularExpression"));
		const char* text = pins.GetPinData<const char>(NOS_NAME("Input"));
		std::string inputCopy = text;

		auto& outMatchPin = pins[NOS_NAME("Match")].Id;
		auto& outGroupsPin = pins[NOS_NAME("Groups")].Id;

		std::smatch match;
		try
		{
			if (strncmp(Expression.c_str(), expression, std::max(Expression.size(), strlen(expression))) != 0)
			{
				Compiled = std::regex(expression);
				Expression = expression;
			}
			if (std::regex_search(inputCopy, match, Compiled))
				nosEngine.SetPinValue(outMatchPin, nos::Buffer::From(true));
			else
				nosEngine.SetPinValue(outMatchPin, nos::Buffer::From(false));
		}
		catch (std::exception const& e)
		{
			Groups.clear();
			nosEngine.SetPinValue(outMatchPin, nos::Buffer::From(false));
			SetNodeStatusMessage(e.what(), fb::NodeStatusMessageType::FAILURE);
			return NOS_RESULT_SUCCESS;
		}

		std::vector<std::string> curGroups;
		for (size_t i = 0; i < match.size(); ++i)
		{
			curGroups.push_back(match[i]);
		}
		if (Groups != curGroups)
		{
			Groups = curGroups;
			std::vector<fb::TNodeStatusMessage> messages;
			for (size_t i = 0; i < curGroups.size(); ++i)
			{
				std::string groupStr = "Group " + std::to_string(i) + ": " + curGroups[i];
				messages.push_back({{}, groupStr, fb::NodeStatusMessageType::INFO });
			}
			SetNodeStatusMessages(messages);
		}

		if (!curGroups.empty())
		{
			flatbuffers::FlatBufferBuilder fbb;
			std::vector<flatbuffers::Offset<flatbuffers::String>> elements;
			for (int i = 0; i < curGroups.size(); ++i)
				elements.push_back(fbb.CreateString(curGroups[i].c_str()));
			fbb.Finish(fbb.CreateVector(elements));
			nos::Buffer groupsBuf = fbb.Release();
			auto root = flatbuffers::GetRoot<uint8_t>(groupsBuf.Data()); // TODO: Move array creation code to API.
			nosEngine.SetPinValue(outGroupsPin, { (void*)root, groupsBuf.Size() });
		}
		else
		{
			ClearNodeStatusMessages();
		}

		return NOS_RESULT_SUCCESS;
	}

	std::vector<std::string> Groups;
	std::regex Compiled;
	std::string Expression;
};

nosResult RegisterRegex(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Regex"), RegexNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::strings