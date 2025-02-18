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
					AddFbPin(pin);
		}
	}

	nos::uuid SwitchFuncId;
	struct CasePinInfo
	{
		std::optional<int> CaseNum = std::nullopt;
		bool IsDuplicate = false;
	};
	std::unordered_map<nos::uuid, CasePinInfo> FuncPinIdToCaseMap;

	void OnFunctionUpdated(nosNodeFunctionUpdate const* update) override
	{
		if (update->FunctionName != NSN_Switch)
			return;
		if (update->Type == NOS_NODE_FUNCTION_UPDATE_NODE_UPDATE)
		{
			auto& nodeUpdate = *update->NodeUpdate;
			switch (nodeUpdate.Type)
			{
			case NOS_NODE_UPDATE_PIN_CREATED: 
			{
				nosFbPinPtr pin = nodeUpdate.PinCreated;
				if (pin->show_as() == fb::ShowAs::OUTPUT_PIN)
					AddFbPin(pin);
				break;
			}
			case NOS_NODE_UPDATE_PIN_DELETED: FuncPinIdToCaseMap.erase(nodeUpdate.PinDeleted); break;
			}
		}
		else
		{
			auto& pinUpdate = *update->PinUpdate;
			if (!FuncPinIdToCaseMap.contains(pinUpdate.PinId))
				return;
			if (pinUpdate.UpdatedField == NOS_PIN_FIELD_DISPLAY_NAME)
			{
				auto caseNum = DisplayNameToCase(nos::Name(pinUpdate.DisplayName).AsString());
				auto oldCaseInfo = FuncPinIdToCaseMap[pinUpdate.PinId];
				if (oldCaseInfo.CaseNum == caseNum)
					return;
				FuncPinIdToCaseMap.erase(pinUpdate.PinId);
				if (oldCaseInfo.CaseNum && !oldCaseInfo.IsDuplicate)
					for (auto& [id, otherCaseInfo] : FuncPinIdToCaseMap)
					{
						if (otherCaseInfo.CaseNum == oldCaseInfo.CaseNum)
						{
							otherCaseInfo.IsDuplicate = false;
							UpdatePinMetadataAndOrphan(id, otherCaseInfo);
						}
					}
				AddCasePinInfo(pinUpdate.PinId, caseNum);
			}
		}
	}

	void AddFbPin(nosFbPinPtr pin)
	{
		std::string dispName;
		if (auto pinDispName = pin->display_name())
			dispName = pinDispName->str();
		else
			dispName = pin->name()->str();

		AddCasePinInfo(*pin->id(), DisplayNameToCase(dispName), pin->meta_data_map()->LookupByKey("Duplicate"));
	}

	void AddCasePinInfo(nos::uuid const& pinId,
						std::optional<int> caseNum,
						std::optional<bool> isDuplicate = std::nullopt)
	{
		CasePinInfo caseInfo{.CaseNum = caseNum, .IsDuplicate = isDuplicate.value_or(false)};
		if (caseInfo.CaseNum && !caseInfo.IsDuplicate)
			for (auto const& [_, otherCaseInfo] : FuncPinIdToCaseMap)
				if (!otherCaseInfo.IsDuplicate && otherCaseInfo.CaseNum && *otherCaseInfo.CaseNum == *caseInfo.CaseNum)
				{
					caseInfo.IsDuplicate = true;
					break;
				}
		FuncPinIdToCaseMap[pinId] = caseInfo;
		UpdatePinMetadataAndOrphan(pinId, caseInfo);
	}

	void UpdatePinMetadataAndOrphan(nos::uuid const& pinId, CasePinInfo const& caseInfo)
	{
		TPartialPinUpdate update;
		update.pin_id = pinId;
		update.orphan_state = std::make_unique<fb::TPinOrphanState>();
		update.orphan_state->type =
			caseInfo.CaseNum && !caseInfo.IsDuplicate ? fb::PinOrphanStateType::ACTIVE : fb::PinOrphanStateType::ORPHAN;
		if (!caseInfo.CaseNum)
			update.orphan_state->message = "Pin name is invalid. Name must be in format: `Name Number`.";
		else if (caseInfo.IsDuplicate)
			update.orphan_state->message = "Case number is duplicated.";
		update.clear_metadata = true;
		if (caseInfo.IsDuplicate)
		{
			auto& metadata = update.meta_data_map.emplace_back(std::make_unique<fb::TMetaDataEntry>());
			metadata->key = "Duplicate";
			metadata->value = "true";
		}
		flatbuffers::FlatBufferBuilder fbb;
		HandleEvent(CreateAppEvent(fbb, nos::CreatePartialPinUpdate(fbb, &update)));
	}

	std::optional<int> DisplayNameToCase(std::string const& name)
	{
		auto pos = name.find_last_of(' ');
		if (pos == std::string::npos)
			pos = 0;
		else
			pos++;
		auto num = name.substr(pos);
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
			for (auto& [_, caseInfo] : FuncPinIdToCaseMap)
				if (caseInfo.CaseNum && *caseInfo.CaseNum > maxCaseNum)
					maxCaseNum = *caseInfo.CaseNum;
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
		for (auto& [pinId, pinCaseInfo] : FuncPinIdToCaseMap)
		{
			if (!pinCaseInfo.CaseNum || pinCaseInfo.IsDuplicate)
				continue;
			if (caseNum != *pinCaseInfo.CaseNum)
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