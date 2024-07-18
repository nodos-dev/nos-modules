// Copyright MediaZ Teknoloji A.S. All Rights Reserved.
#include <Nodos/SubsystemAPI.h>
#include <Nodos/Helpers.hpp>

#include <nosPython/Python_generated.h>

// External
#define PYBIND11_SIMPLE_GIL_MANAGEMENT
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/embed.h>
namespace pyb = pybind11;
using namespace pyb::literals;

NOS_INIT()

namespace nos::py
{
class Interpreter
{
public:
	Interpreter() : ScopedInterpreter(), Release()
	{
		pyb::gil_scoped_acquire gil;
		auto sysInfo = pyb::module::import("sys").attr("version").cast<std::string>();
		nosEngine.LogI("%s", sysInfo.c_str());
		try {
			NodosInternalModule = std::make_unique<pyb::module>(pyb::module::import("__nodos_internal__"));
		}
		catch (std::exception& e)
		{
			nosEngine.LogE(e.what(), "Failed to import __nodos_internal__ with error %s", e.what());
		}
	}

	~Interpreter()
	{
		NodeInstances.clear();
		NodosInternalModule.reset();
	}

	void AddPath(std::string path)
	{
		nosEngine.LogD("Adding module search path: %s", path.c_str());
		pyb::module::import("sys").attr("path").attr("append")(path);
	}

	nosResult ImportModule(nos::Name name, std::filesystem::path pySourcePath) {
		if (Modules.contains(name)) {
			try {
				pyb::gil_scoped_acquire gil;
				Modules[name]->reload();
			}
			catch (std::exception& e)
			{
				nosEngine.LogE(e.what(), "Failed to reload module %s with error %s", name.AsCStr(), e.what());
				return NOS_RESULT_FAILED;
			}
			return NOS_RESULT_SUCCESS;
		}
		try {
			pyb::gil_scoped_acquire gil;
			AddPath(pySourcePath.parent_path().generic_string());
			pyb::module python_module = pyb::module::import(pySourcePath.filename().stem().generic_string().c_str());
			Modules[name] = std::make_unique<pyb::module>(std::move(python_module));
		}
		catch (std::exception& e)
		{
			nosEngine.LogE(e.what(), "Failed to import module %s with error %s", name.AsCStr(), e.what());
			return NOS_RESULT_FAILED;
		}
		return NOS_RESULT_SUCCESS;
	}

	void CreateNodeInstance(nos::fb::UUID id, nos::Name className,nos::Name name)
	{
		try {
			pyb::gil_scoped_acquire gil;
			pyb::object python_object = Modules[className]->attr(name.AsCStr());

			pyb::object instance = python_object();
			NodeInstances[id] = std::make_unique<pyb::object>(std::move(instance));
		}
		catch (std::exception& e)
		{
			nosEngine.LogE(e.what(), "Failed to create instance of %s with error %s", name.AsCStr(), e.what());
		}
	}

	void RemoveNodeInstance(nos::fb::UUID id)
	{
		if (NodeInstances.contains(id)) {
			pyb::gil_scoped_acquire gil;
			NodeInstances.erase(id);
		}
	}

	std::shared_ptr<pyb::object> GetNodeInstance(nos::fb::UUID id)
	{
		auto it = NodeInstances.find(id);
		if (it == NodeInstances.end())
			return nullptr;
		return it->second;
	}

protected:
	pyb::scoped_interpreter ScopedInterpreter;
	pyb::gil_scoped_release Release;
	std::unordered_map<nos::fb::UUID, std::shared_ptr<pyb::object>> NodeInstances;
	std::unordered_map<nos::Name, std::unique_ptr<pyb::module>> Modules;
	std::unique_ptr<pyb::module> NodosInternalModule;
};

static Interpreter* GInterpreter = nullptr;
static std::vector<std::unique_ptr<nos::fb::UUID>> NodeUUIDs;

// Classes with prefix 'PyNative' are only constructed from C++ side, with 'Py' are for Python-constructible.
class PyNativeNodeExecuteArgs : public nos::NodeExecuteArgs
{
public:
	using nos::NodeExecuteArgs::NodeExecuteArgs;

	std::optional<pyb::memoryview> GetPinValue(std::string pinName) const
	{
		auto it = this->find(nos::Name(pinName));
		if (it == this->end())
			return std::nullopt;
		auto buf = it->second.Data;
		return pyb::memoryview::from_memory(buf->Data, buf->Size);
	}
	std::optional<nosUUID> GetPinId(std::string pinName) const
	{
		auto it = this->find(nos::Name(pinName));
		if (it == this->end())
			return std::nullopt;
		return it->second.Id;
	}
};

class PyNativeOnPinValueChangedArgs {
public:
	PyNativeOnPinValueChangedArgs() : PinName(), Value() {}
	nos::Name PinName = {};
	nosBuffer Value = {};

	pyb::memoryview GetPinValue() const
	{
		return pyb::memoryview::from_memory(Value.Data, Value.Size);
	}
	nos::Name GetPinName() const
	{
		return PinName;
	}
};

class PyNativeOnPinConnectedArgs {
public:
	PyNativeOnPinConnectedArgs() : PinName() {}
	nos::Name PinName = {};
};

class PyNativeOnPinDisconnectedArgs {
public:
	PyNativeOnPinDisconnectedArgs() : PinName() {}
	nos::Name PinName = {};
};

PYBIND11_EMBEDDED_MODULE(__nodos_internal__, m)
{
	// Enums
	pyb::class_<nosResult>(m, "result")
		.def_property_readonly_static("SUCCESS", [](const pyb::object&) {return NOS_RESULT_SUCCESS; })
		.def_property_readonly_static("FAILED", [](const pyb::object&) {return NOS_RESULT_FAILED; });

	// Structs
	pyb::class_<PyNativeNodeExecuteArgs>(m, "NodeExecuteArgs")
		.def_property_readonly("node_class_name", [](const PyNativeNodeExecuteArgs& args) -> std::string_view { return args.NodeClassName.AsCStr(); })
		.def_property_readonly("node_name", [](const PyNativeNodeExecuteArgs& args) -> std::string_view { return args.NodeName.AsCStr(); })
		.def("get_pin_value", &PyNativeNodeExecuteArgs::GetPinValue, "Access the memory of the pin specified by 'pin_name'", "pin_name"_a)
		.def("get_pin_id", &PyNativeNodeExecuteArgs::GetPinId, "Get the unique identifier of the pin specified by 'pin_name'", "pin_name"_a);

	pyb::class_<PyNativeOnPinValueChangedArgs>(m, "PinValueChangedArgs")
		.def_property_readonly("pin_value", [](const PyNativeOnPinValueChangedArgs& args) -> pyb::memoryview { return args.GetPinValue(); })
		.def_property_readonly("pin_name", [](const PyNativeOnPinValueChangedArgs& args) -> std::string_view { return args.PinName.AsCStr(); });

	pyb::class_<PyNativeOnPinConnectedArgs>(m, "PinConnectedArgs")
		.def_property_readonly("pin_value", [](const PyNativeOnPinConnectedArgs& args) -> std::string_view { return args.PinName.AsCStr(); });
	
	pyb::class_<PyNativeOnPinDisconnectedArgs>(m, "PinDisconnectedArgs")
		.def_property_readonly("pin_value", [](const PyNativeOnPinDisconnectedArgs& args) -> std::string_view { return args.PinName.AsCStr(); });

	pyb::class_<nosUUID>(m, "uuid")
		.def(pyb::init<>())
		.def("__str__", [](const nosUUID& id) -> std::string { return nos::UUID2STR(id); })
		.def("__hash__", [](const nosUUID& id) -> size_t { return nos::UUIDHash(id); })
		.def("__eq__", [](const nosUUID& self, const nosUUID& other) -> bool { return self == other; });

	// Engine Services
	m.def("set_pin_value",
		[](const nosUUID& id, const pyb::buffer& buf) {
			auto info = buf.request();
			nosEngine.SetPinValue(id, nosBuffer{.Data = info.ptr, .Size = info.size < 0 ? 0ull : (size_t(info.size) * info.itemsize)});
		});
	m.def("log_info",
		[](const std::string& log) {
			nosEngine.LogI(log.c_str());
		});
	m.def("log_warning",
		[](const std::string& log) {
			nosEngine.LogW(log.c_str());
		});
	m.def("log_error",
		[](const std::string& log) {
			nosEngine.LogE(log.c_str());
		});
}

void Init()
{
	GInterpreter = new Interpreter;
	return;
}

void Deinit()
{
	delete GInterpreter;
}

nosResult NOSAPI_CALL OnPyNodeRegistered(nosModuleIdentifier pluginId, nosName className, nosBuffer options)
{
	char path[2048];
	nosEngine.GetModuleFolderPath(pluginId, 2048, path);
	fs::path moduleRoot = std::string(path);

	auto* pyNodeOptions = flatbuffers::GetRoot<PythonNode>(options.Data);

	if (!flatbuffers::IsFieldPresent(pyNodeOptions, PythonNode::VT_SOURCE))
		return NOS_RESULT_INVALID_ARGUMENT;

	fs::path relSourcePath = pyNodeOptions->source()->str();
	std::string sourcePathStr = (moduleRoot / relSourcePath).generic_string();

	if (!fs::exists(sourcePathStr))
	{
		nosEngine.LogE("Python Subsystem: Source file %s does not exist.", sourcePathStr.c_str());
		return NOS_RESULT_INVALID_ARGUMENT;
	}

	return GInterpreter->ImportModule(className, std::filesystem::canonical(sourcePathStr));
}

void NOSAPI_CALL OnNodeCreated(const nosFbNode* node, void** outCtxPtr) {
	GInterpreter->CreateNodeInstance(*node->id(), nos::Name(node->class_name()->str()),nos::Name(node->name()->str()));
	NodeUUIDs.push_back(std::make_unique<nos::fb::UUID>(*node->id()));
	(*outCtxPtr) = NodeUUIDs[NodeUUIDs.size() - 1].get();
}

void NOSAPI_CALL OnNodeDeleted(void* ctx, nosUUID nodeId) {

	GInterpreter->RemoveNodeInstance(nodeId);
}

void NOSAPI_CALL OnPinValueChanged(void* ctx, nosName pinName, nosUUID pinId, nosBuffer value)
{
	pyb::gil_scoped_acquire gil;
	nos::fb::UUID id = *static_cast<nos::fb::UUID*>(ctx);
	PyNativeOnPinValueChangedArgs args;
	args.PinName = nos::Name(pinName);
	args.Value = value;
	GInterpreter->GetNodeInstance(id)->attr("on_pin_value_changed")(args);
}

void NOSAPI_CALL OnPinConnected(void* ctx, nosName pinName, nosUUID connectedPin, nosUUID nodeId)
{
	pyb::gil_scoped_acquire gil;
	nos::fb::UUID id = *static_cast<nos::fb::UUID*>(ctx);
	PyNativeOnPinConnectedArgs args;
	args.PinName = nos::Name(pinName);
	GInterpreter->GetNodeInstance(id)->attr("on_pin_connected")(args);
}

void NOSAPI_CALL OnPinDisconnected(void* ctx, nosName pinName)
{
	pyb::gil_scoped_acquire gil;
	nos::fb::UUID id = *static_cast<nos::fb::UUID*>(ctx);
	PyNativeOnPinDisconnectedArgs args;
	args.PinName = nos::Name(pinName);
	GInterpreter->GetNodeInstance(id)->attr("on_pin_disconnected")(args);
}

nosResult NOSAPI_CALL ExecutePyNode(void* ctx, const nosNodeExecuteArgs* args)
{
	auto m = GInterpreter->GetNodeInstance(args->NodeId);
	if (!m)
		return NOS_RESULT_NOT_FOUND;

	try {
		pyb::gil_scoped_acquire gil;
		PyNativeNodeExecuteArgs pyNativeArgs(args);
		auto ret = static_cast<nosResult>(m->attr("execute_node")(pyNativeArgs).cast<int>());
		return ret;
	}
	catch (std::exception& exp)
	{
		nosEngine.LogDE(exp.what(), "%s execution failed. See details.", nos::Name(args->NodeClassName).AsCStr());
		return NOS_RESULT_FAILED;
	}
	return NOS_RESULT_SUCCESS;
}
} // namespace nos::py

extern "C"
{

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystem(nosSubsystemFunctions* subsystemFunctions)
{
	nos::py::Init();
	
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR nosResult NOSAPI_CALL nosUnloadSubsystem()
{
	nos::py::Deinit();
	// Python DLL might not be released when nos.py is unloaded, due to some third party python module (like numpy).
	// Since PyFinalize is called, interpreter ends up in an invalid state and
	// subsequent imports after nos.py reloaded will cause Nodos to crash.
	// Related issues:
	// - https://github.com/numpy/numpy/issues/8097
	// - https://github.com/python/cpython/issues/78490
	// - https://bugs.python.org/issue401713#msg34524
	// One solution would be to statically link Python to nos.py & fork Python to solve the
	// issue (or create a PR and wait for them to add to the release).
	// Another definite & future-proof solution would be to spin-up a new process that embeds Python and communicate it
	// via IPC from nos.py. But this creates some limitations & complexity.
	return NOS_RESULT_SUCCESS;
}

NOSAPI_ATTR nosResult NOSAPI_CALL nosExportSubsystemNodeFunctions(size_t* outSize, nosSubsystemNodeFunctions** outList)
{
	*outSize = 1;
	if (!outList)
		return NOS_RESULT_SUCCESS;
	auto pyNodeFunctions = outList[0];
	pyNodeFunctions->OnNodeClassRegistered = nos::py::OnPyNodeRegistered;
	pyNodeFunctions->NodeType = NOS_NAME_STATIC("nos.py.PythonNode");
	pyNodeFunctions->NodeFunctions.OnNodeCreated = nos::py::OnNodeCreated;
	pyNodeFunctions->NodeFunctions.OnNodeDeleted = nos::py::OnNodeDeleted;
	pyNodeFunctions->NodeFunctions.ExecuteNode = nos::py::ExecutePyNode;
	pyNodeFunctions->NodeFunctions.OnPinValueChanged = nos::py::OnPinValueChanged;
	pyNodeFunctions->NodeFunctions.OnPinConnected = nos::py::OnPinConnected;
	pyNodeFunctions->NodeFunctions.OnPinDisconnected = nos::py::OnPinDisconnected;
	
	return NOS_RESULT_SUCCESS;
}

}