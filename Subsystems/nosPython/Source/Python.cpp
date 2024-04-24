// Copyright Nodos AS. All Rights Reserved.
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
		NodosModule = std::make_unique<pyb::module>(pyb::module::import("nodos"));
	}

	~Interpreter()
	{
		pyb::gil_scoped_acquire gil;
		Modules.clear();
		NodosModule.reset();
	}

	void AddPath(std::string path)
	{
		nosEngine.LogD("Adding module search path: %s", path.c_str());
		pyb::module::import("sys").attr("path").attr("append")(path);
	}

	void Import(nos::Name nodeClassName, std::filesystem::path pySourcePath)
	{
		pyb::gil_scoped_acquire gil;
		if (auto m = GetPyModule(nodeClassName))
		{
			nosEngine.LogI("Reloading %s source", nodeClassName.AsCStr());
			m->reload();
			return;
		}
		nosEngine.LogI("Importing %s source from %s", nodeClassName.AsCStr(), pySourcePath.generic_string().c_str());
		AddPath(pySourcePath.parent_path().generic_string());
		auto imported = pyb::module::import(pySourcePath.filename().stem().generic_string().c_str());
		Modules[nodeClassName] = std::make_unique<pyb::module>(std::move(imported));
	}

	pyb::module* GetPyModule(nosName nodeClassName)
	{
		auto it = Modules.find(nodeClassName);
		if (it == Modules.end())
			return nullptr;
		return it->second.get();
	}

protected:
	pyb::scoped_interpreter ScopedInterpreter;
	pyb::gil_scoped_release Release;
	std::unordered_map<nos::Name, std::unique_ptr<pyb::module>> Modules;
	std::unique_ptr<pyb::module> NodosModule;
};

static Interpreter* GInterpreter = nullptr;

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

PYBIND11_EMBEDDED_MODULE(nodos, m)
{
	// Enums
	pyb::class_<nosResult>(m, "result")
		.def_property_readonly_static("SUCCESS", [](const pyb::object&) {return NOS_RESULT_SUCCESS; })
		.def_property_readonly_static("FAILED", [](const pyb::object&) {return NOS_RESULT_FAILED; });

	// Structs
	pyb::class_<PyNativeNodeExecuteArgs>(m, "NodeExecuteArgs")
		.def_property_readonly("node_class_name", [](const PyNativeNodeExecuteArgs& args) -> std::string_view { return args.NodeClassName.AsCStr(); })
		.def_property_readonly("node_name", [](const PyNativeNodeExecuteArgs& args) -> std::string_view { return args.NodeName.AsCStr(); })
		.def("pin_value", &PyNativeNodeExecuteArgs::GetPinValue, "Access the memory of the pin specified by 'pin_name'", "pin_name"_a)
		.def("pin_id", &PyNativeNodeExecuteArgs::GetPinId, "Get the unique identifier of the pin specified by 'pin_name'", "pin_name"_a);

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
	
	try
	{
		GInterpreter->Import(className, std::filesystem::canonical(sourcePathStr));
	}
	catch (std::exception& e)
	{
		nosEngine.LogDE(e.what(), "Failed to import module %s. See details.", sourcePathStr.c_str());
		return NOS_RESULT_FAILED;
	}
	return NOS_RESULT_SUCCESS;
}

nosResult NOSAPI_CALL ExecutePyNode(void* ctx, const nosNodeExecuteArgs* args)
{
	auto m = GInterpreter->GetPyModule(args->NodeClassName);
	if (!m)
		return NOS_RESULT_NOT_FOUND;
	
	try {
		pyb::gil_scoped_acquire gil;
		PyNativeNodeExecuteArgs pyNativeArgs(args);
		auto ret = m->attr("execute_node")(pyNativeArgs).cast<nosResult>();
		return ret;
	}
	catch (std::exception& exp)
	{
		nosEngine.LogDE(exp.what(), "%s execution failed. See details.", nos::Name(args->NodeClassName).AsCStr());
		return NOS_RESULT_SUCCESS;
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
	pyNodeFunctions->NodeType = NOS_NAME_STATIC("nos.py.PythonNode");
	pyNodeFunctions->OnNodeClassRegistered = nos::py::OnPyNodeRegistered;
	pyNodeFunctions->ExecuteNode = nos::py::ExecutePyNode;
	return NOS_RESULT_SUCCESS;
}

}