#include <Nodos/PluginHelpers.hpp>


namespace nos::utilities
{
NOS_REGISTER_NAME(Switch)
struct SwitchTrigger : NodeContext
{
	SwitchTrigger(nosFbNodePtr inNode) : NodeContext(inNode) 
	{
		for (auto* func : *inNode->functions())
		{
			if (NSN_Switch != func->name()->string_view())
				continue;
			SwitchFuncId = *func->id();
			for (auto* pin : *func->pins())
				if (pin->show_as() == fb::ShowAs::OUTPUT_PIN)
					if (AddPin(pin))
						SetPinOrphanState(*pin->id(), fb::PinOrphanStateType::ACTIVE);
		}
	}
	nos::uuid SwitchFuncId;
	std::unordered_map<nos::uuid, std::optional<int>> FuncPinIdToCaseMap;

	void OnNodeUpdated(nosNodeUpdate const* update) override
	{
		if (update->Type != NOS_NODE_UPDATE_FUNCTION_UPDATED)
			return;
		auto& funcUpdate = update->FunctionUpdated;
		if (funcUpdate->FunctionName != NSN_Switch)
			return;
		if (funcUpdate->IsNodeUpdate)
		{
			auto& nodeUpdate = *funcUpdate->NodeUpdate;
			switch (nodeUpdate.Type)
			{
			case NOS_NODE_UPDATE_PIN_CREATED: 
			{
				nosFbPinPtr pin = nodeUpdate.PinCreated;
				if (pin->show_as() == fb::ShowAs::OUTPUT_PIN)
					AddPin(pin);
				break;
			}
			case NOS_NODE_UPDATE_PIN_DELETED: FuncPinIdToCaseMap.erase(nodeUpdate.PinDeleted); break;
			}
		}
		else
		{
			auto& pinUpdate = *funcUpdate->PinUpdate;
			if (!FuncPinIdToCaseMap.contains(pinUpdate.PinId))
				return;
			if (pinUpdate.UpdatedField == NOS_PIN_FIELD_DISPLAY_NAME)
			{
				auto caseNum = DisplayNameToCase(nos::Name(pinUpdate.DisplayName).AsString());
				FuncPinIdToCaseMap[pinUpdate.PinId] = caseNum;
				if (caseNum)
					SetPinOrphanState(pinUpdate.PinId, fb::PinOrphanStateType::ACTIVE);
				else
					SetPinOrphanState(pinUpdate.PinId,
									  fb::PinOrphanStateType::ORPHAN,
									  "Pin name is invalid. Name must be in format: `Name Number`.");
			}
		}
	}

	bool AddPin(nosFbPinPtr pin)
	{
		std::string dispName;
		if (auto pinDispName = pin->display_name())
			dispName = pinDispName->str();
		else
			dispName = pin->name()->str();
		auto caseNum = DisplayNameToCase(dispName);	
		FuncPinIdToCaseMap[*pin->id()] = caseNum;
		if (!caseNum)
			SetPinOrphanState(*pin->id(),
							  fb::PinOrphanStateType::ORPHAN,
							  "Pin name is invalid. Name must be in format: `Name Number`.");
		return caseNum.has_value();
	}

	std::optional<int> DisplayNameToCase(std::string const& name)
	{
		auto pos = name.find_last_of(' ');
		if (pos == std::string::npos)
			return std::nullopt;
		auto num = name.substr(pos + 1);
		if (num.empty())
			return std::nullopt;
		try
		{
			return std::stoi(num);
		}
		catch (std::exception const&)
		{
			return std::nullopt;
		}
	}

	void OnMenuRequested(nosContextMenuRequestPtr request) override
	{
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<flatbuffers::Offset<nos::ContextMenuItem>> items;
		if (*request->item_id() == NodeId)
			items.push_back(nos::CreateContextMenuItemDirect(fbb, "Add Case", 1));
		else
		{
			if (!FuncPinIdToCaseMap.contains(*request->item_id()) || FuncPinIdToCaseMap.size() <= 1)
				return;
			items.push_back(nos::CreateContextMenuItemDirect(fbb, "Remove Output", 1));
		}

		auto event = CreateAppEvent(
			fbb,
			CreateAppContextMenuUpdate(
				fbb, request->item_id(), request->pos(), request->instigator(), fbb.CreateVector(items)));

		HandleEvent(event);
	}

	void OnMenuCommand(uuid const& itemID, uint32_t cmd) override
	{
		flatbuffers::FlatBufferBuilder fbb;
		if (itemID == NodeId)
		{
			int maxCaseNum = 0;
			for (auto& [_, caseNum] : FuncPinIdToCaseMap)
				if (caseNum && *caseNum > maxCaseNum)
					maxCaseNum = *caseNum;
			maxCaseNum++;
			fb::TPin pin{};
			pin.id = uuid(nosEngine.GenerateID());
			pin.name = "Case_" + std::to_string(maxCaseNum);
			pin.display_name = "Case " + std::to_string(maxCaseNum);
			pin.type_name = nos::exe::GetFullyQualifiedName();
			pin.show_as = fb::ShowAs::OUTPUT_PIN;
			pin.can_show_as = fb::CanShowAs::OUTPUT_PIN_ONLY;
			nos::TPartialNodeUpdate update;
			update.node_id = SwitchFuncId;
			update.pins_to_add.emplace_back(std::make_unique<fb::TPin>(std::move(pin)));
			HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdate(fbb, &update)));
		}
		else
		{
			if (FuncPinIdToCaseMap.size() <= 1 || !FuncPinIdToCaseMap.contains(itemID))
				return;
			nos::TPartialNodeUpdate update;
			update.node_id = SwitchFuncId;
			update.pins_to_delete = {itemID};
			HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdate(fbb, &update)));
		}
	}

	nosResult Switch(nosFunctionExecuteParams* functionExecParams)
	{
		NodeExecuteParams params(functionExecParams->FunctionNodeExecuteParams);
		int caseNum = *params.GetPinData<int>(NOS_NAME("Case"));
		functionExecParams->MarkOutExeDirty = false;
		for (auto& [pinId, pinCaseNum] : FuncPinIdToCaseMap)
		{
			if (!pinCaseNum)
				continue;
			if (caseNum != *pinCaseNum)
				continue;
			nosEngine.SetPinDirty(pinId);
		}
		return NOS_RESULT_SUCCESS;
	}

	NOS_DECLARE_FUNCTIONS(
		NOS_ADD_FUNCTION(NOS_NAME("Switch"), Switch),
	);
};


nosResult RegisterSwitchTrigger(nosNodeFunctions* out)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("SwitchTrigger"), SwitchTrigger, out);
	return NOS_RESULT_SUCCESS;
}

}