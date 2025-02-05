// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#pragma once

#include <Nodos/PluginHelpers.hpp>

namespace nos::utilities
{



struct SyncMultiOutletNode : NodeContext
{
	struct SyncOutPin
	{
		std::shared_mutex Mutex;
		uint64_t ConnectedPinCount = 0;
		uint64_t ProcessedFrameNumber = 0;
	};

	SyncMultiOutletNode(nosFbNode const* node) : NodeContext(node), InputPin{}
	{ 
		for (auto* pin : *node->pins())
			if (pin->show_as() == nos::fb::ShowAs::OUTPUT_PIN)
			{
				SetPinOrphanState(*pin->id(), nos::fb::PinOrphanStateType::ACTIVE);
				OutPins[nosUUID(*pin->id())];
				OutPinIdsOrdered.push_back(nosUUID(*pin->id()));
			}
	}

	void OnPartialNodeUpdated(nosNodeUpdate const* update) override
	{
		if (update->Type == NOS_NODE_UPDATE_PIN_CREATED)
		{
			std::unique_lock lock(OutPinsMutex);
			OutPins[nosUUID(*update->PinCreated->id())];
			OutPinIdsOrdered.push_back(nosUUID(*update->PinCreated->id()));
		}
		else if (update->Type == NOS_NODE_UPDATE_PIN_DELETED)
		{
			{
				std::unique_lock lock(OutPinsMutex);
				OutPins.erase(update->PinDeleted);
				std::erase(OutPinIdsOrdered, update->PinDeleted);
			}
			if (RequestNewInput())
			{
				nosScheduleNodeParams params = {.NodeId = NodeId, .AddScheduleCount = 1};
				nosEngine.ScheduleNode(&params);
			}
		}
	}
	
	std::atomic_bool Exit = true;
	struct SyncInPin
	{
		std::mutex Mutex;
		std::condition_variable CV;
		const nosBuffer* Data = nullptr;
		uint64_t FrameNumber = 0;
		bool Requested = false;
	} InputPin;

	std::shared_mutex OutPinsMutex;
	std::unordered_map<nosUUID, SyncOutPin> OutPins;
	std::vector<nosUUID> OutPinIdsOrdered;

	enum WaitResult
	{
		OK,
		EXIT,
		TIMEOUT
	};

	WaitResult WaitInput(uint64_t reqFrameNumber)
	{
		std::unique_lock lock(InputPin.Mutex);
		if (!InputPin.CV.wait_for(
				lock, std::chrono::milliseconds(100), [&] { return InputPin.FrameNumber >= reqFrameNumber || Exit; }))
			return TIMEOUT;
		assert(Exit || InputPin.FrameNumber == reqFrameNumber);
		if (Exit)
			return EXIT;
		return OK;
	}

	bool RequestNewInput()
	{
		std::unique_lock inputLock(InputPin.Mutex);
		uint64_t inFrameNumber = InputPin.FrameNumber;
		std::shared_lock outPinsLock(OutPinsMutex);
		for (auto& [name, pin] : OutPins)
		{
			std::shared_lock outputLock(pin.Mutex);
			if (pin.ConnectedPinCount == 0)
				continue;
			if (pin.ProcessedFrameNumber < inFrameNumber)
				return false;
		}
		if (InputPin.Requested)
			return false;
		InputPin.Requested = true;
		return true;
	}

	nosResult CopyFrom(nosCopyInfo* cpy) override
	{
		std::shared_lock outPinsLock(OutPinsMutex);
		auto& outPin = OutPins[cpy->ID];
		std::shared_lock lock(outPin.Mutex);
		uint64_t lastOutFrameNum = outPin.ProcessedFrameNumber;
		if (auto res = WaitInput(lastOutFrameNum + 1))
			return res == EXIT ? NOS_RESULT_FAILED : NOS_RESULT_PENDING;
		// This place is ok to not lock the input pin mutex, because we are sure that the input pin is not being written
		// to since at least this out pin is still processing the frame.
		outPin.ProcessedFrameNumber = InputPin.FrameNumber;
		cpy->CopyFromOptions.ShouldSetSourceFrameNumber = true;
		cpy->FrameNumber = InputPin.FrameNumber;

		if (RequestNewInput())
		{
			nosScheduleNodeParams params = {.NodeId = NodeId, .AddScheduleCount = 1};
			nosEngine.ScheduleNode(&params);
		}
		return NOS_RESULT_SUCCESS;
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		nos::NodeExecuteParams args(params);
		assert(InputPin.Requested);
		InputPin.Data = args[NOS_NAME("Input")].Data;
		std::unique_lock lock(InputPin.Mutex);
		++InputPin.FrameNumber;
		InputPin.Requested = false;
		InputPin.CV.notify_all();
		return NOS_RESULT_SUCCESS;
	}

	void OnPathStop() override
	{ 
		Exit = true;
		InputPin.CV.notify_all();
	}

	void OnPathStart() override
	{
		Exit = false;
		for (auto& [name, pin] : OutPins)
			pin.ProcessedFrameNumber = 0;
		InputPin.FrameNumber = 0;
		InputPin.Requested = true;
		nosScheduleNodeParams params = {.NodeId = NodeId, .AddScheduleCount = 1};
		nosEngine.ScheduleNode(&params);
	}

	void OnPinConnected(nos::Name pinName, nosUUID) override 
	{
		auto& pin = *GetPin(pinName);
		if (pin.ShowAs != nosFbShowAs::OUTPUT_PIN)
			return;
		std::shared_lock outPinsLock(OutPinsMutex);
		auto& outPin = OutPins[pin.Id];
		std::unique_lock lock(outPin.Mutex);
		outPin.ConnectedPinCount++;
	}

	void OnPinDisconnected(nos::Name pinName) override 
	{
		auto& pin = *GetPin(pinName);
		if (pin.ShowAs != nosFbShowAs::OUTPUT_PIN)
			return;
		std::shared_lock outPinsLock(OutPinsMutex);
		auto& outPin = OutPins[pin.Id];
		{
			std::unique_lock lock(outPin.Mutex);
			outPin.ConnectedPinCount--;
		}
	}

	void OnMenuRequested(const nosContextMenuRequest* request) override
	{
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<flatbuffers::Offset<nos::ContextMenuItem>> items;
		if (*request->item_id() == NodeId)
			items.push_back(nos::CreateContextMenuItemDirect(fbb, "Add Output", 1));
		else
		{
			auto& pin = *GetPin(*request->item_id());
			if (pin.ShowAs != nosFbShowAs::OUTPUT_PIN)
				return;
			if (pin.Name == NOS_NAME("Out_0") || pin.Name == NOS_NAME("Out_1"))
				return;
			std::shared_lock outPinsLock(OutPinsMutex);
			items.push_back(nos::CreateContextMenuItemDirect(fbb, "Remove Output", 1));
		}

		auto event = CreateAppEvent(
			fbb,
			CreateAppContextMenuUpdate(
				fbb, request->item_id(), request->pos(), request->instigator(), fbb.CreateVector(items)));

		HandleEvent(event);
	}

	void OnMenuCommand(nosUUID itemID, uint32_t cmd) override
	{
		flatbuffers::FlatBufferBuilder fbb;
		if (itemID == NodeId)
		{
			std::shared_lock outPinsLock(OutPinsMutex);
			auto& lastPin = *GetPin(OutPinIdsOrdered.back());
			std::string lastPinName = lastPin.Name.AsCStr();
			uint64_t lastPinIndex = 0;
			try
			{
				lastPinIndex = std::stoull(lastPinName.substr(lastPinName.find_last_of('_') + 1));
			}
			catch (...)
			{
				nosEngine.LogE("Failed to parse the last output pin index.");
				return;
			}
			fb::TPin pin;
			pin.id = nosEngine.GenerateID();
			pin.name = "Out_" + std::to_string(lastPinIndex + 1);
			pin.display_name = "Out";
			pin.type_name = lastPin.TypeName.AsString();
			pin.live = true;
			pin.show_as = fb::ShowAs::OUTPUT_PIN;
			pin.can_show_as = fb::CanShowAs::OUTPUT_PIN_ONLY;
			nos::TPartialNodeUpdate update;
			update.node_id = NodeId;
			update.pins_to_add.emplace_back(std::make_unique<fb::TPin>(std::move(pin)));
			HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdate(fbb, &update)));
		}
		else
		{
			auto& pin = *GetPin(itemID);
			if (pin.ShowAs != nosFbShowAs::OUTPUT_PIN)
				return;
			std::shared_lock outPinsLock(OutPinsMutex);
			if (OutPins.size() <= 1)
				return;
			nos::TPartialNodeUpdate update;
			update.node_id = NodeId;
			update.pins_to_delete = {itemID};
			HandleEvent(CreateAppEvent(fbb, nos::CreatePartialNodeUpdate(fbb, &update)));
		}
	}

};


nosResult RegisterSyncMultiOutlet(nosNodeFunctions* functions)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("SyncMultiOutlet"), SyncMultiOutletNode, functions)
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities