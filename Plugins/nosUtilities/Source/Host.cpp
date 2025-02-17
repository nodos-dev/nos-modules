// Copyright MediaZ Teknoloji A.S. All Rights Reserved.

#include <Nodos/PluginHelpers.hpp>

#if defined(_WIN32)
#include <WinSock2.h>
#include <ws2tcpip.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#include <time.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#endif

#include <thread>

namespace nos::utilities
{
std::string GetHostName()
{
	std::string hostName;
	char buffer[256];
	if (gethostname(buffer, sizeof(buffer)) != 0) {
		perror("gethostname");
		return "";
	}
	hostName = buffer;
	return hostName;
}

std::string GetIpv4Address()
{
#if defined(_WIN32)
	std::string ip;
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		perror("WSAStartup");
		return "";
	}
	char ac[80];
	if (gethostname(ac, sizeof(ac)) == SOCKET_ERROR) {
		perror("gethostname");
		return "";
	}
	struct hostent* phe = gethostbyname(ac);
	if (phe == 0) {
		perror("gethostbyname");
		return "";
	}
	for (int i = 0; phe->h_addr_list[i] != 0; ++i) {
		struct in_addr addr;
		memcpy(&addr, phe->h_addr_list[i], sizeof(struct in_addr));
		ip = inet_ntoa(addr);
	}
	WSACleanup();
#else
	struct ifaddrs* interfaces = nullptr;
	struct ifaddrs* ifa = nullptr;
	char ip[INET6_ADDRSTRLEN];

	if (getifaddrs(&interfaces) == -1) {
		std::cerr << "Error getting network interfaces" << std::endl;
		return "";
	}

	for (ifa = interfaces; ifa != nullptr; ifa = ifa->ifa_next) {
		if (ifa->ifa_addr == nullptr)
			continue;

		// Check if the interface is an IPv4 or IPv6 address
		if (ifa->ifa_addr->sa_family == AF_INET) {  // IPv4
			void* addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
			inet_ntop(AF_INET, addr, ip, sizeof(ip));
			std::string interface_name = ifa->ifa_name;
			if (interface_name == "lo") // Skip loopback interface
				continue;
			freeifaddrs(interfaces);
			return std::string(ip); // Returns the first interface.
		}
	}

	freeifaddrs(interfaces);
	return "";
#endif
	return ip;
}

long long GetUpTime()
{
#ifdef _WIN32
	// Windows-specific implementation
	return GetTickCount64() / 1000;  // Returns uptime in seconds
#elif __unix__ || __unix || __linux__ || __APPLE__
	// Unix-based (Linux, macOS, etc.) implementation
	struct timespec ts;
	if (clock_gettime(CLOCK_BOOTTIME, &ts) == 0) {
		return ts.tv_sec;
	}
	else {
		return -1;
	}
#else
	return -1;  // Unsupported platform
#endif
}

struct HostInfoPins
{
	uuid HostNamePinId;
	uuid IpAddressPinId;
	uuid UptimeSecondsPinId;
};

struct HostInfo
{
	std::string HostName;
	std::string IpAddress; // TODO: Return a map of interfaces and IPs
	uint32_t UptimeS;

	std::unordered_map<uuid, HostInfoPins> Listeners;

	template <auto PinIdMember, typename T>
	void SetPinValue(HostInfoPins const& pinIds, T const& value)
	{
		if constexpr (std::is_same_v<T, std::string>)
		{
			nos::Buffer buf(value.c_str(), value.size() + 1);
			nosEngine.SetPinValue(pinIds.*PinIdMember, buf);
		}
		else
		{
			auto size = sizeof(value);
			nos::Buffer buf(reinterpret_cast<const char*>(&value), size);
			nosEngine.SetPinValue(pinIds.*PinIdMember, buf);
		}
	}

	template <auto Member, auto PinIdMember, typename T>
	void SetIfChanged(T const& value)
	{
		auto& old = this->*Member;
		if (old != value)
		{
			old = value;
			for (auto& [nodeId, pins] : Listeners)
				SetPinValue<PinIdMember>(pins, value);
		}
	}

	void AddListener(uuid nodeId, HostInfoPins pins)
	{
		Listeners[nodeId] = std::move(pins);
		SetPinValue<&HostInfoPins::HostNamePinId>(pins, HostName);
		SetPinValue<&HostInfoPins::IpAddressPinId>(pins, IpAddress);
		SetPinValue<&HostInfoPins::UptimeSecondsPinId>(pins, UptimeS);
		if (!Listeners.empty() && !Worker.joinable())
		{
			nosEngine.LogD("Host: Starting worker thread");
			Worker = std::jthread([this](std::stop_token stopToken) // TODO: Use OS callbacks to subscribe to changes.
			{
				while (!stopToken.stop_requested())
				{
					auto hostName = GetHostName();
					auto ip = GetIpv4Address();
					auto uptime = GetUpTime();
					SetIfChanged<&HostInfo::HostName, &HostInfoPins::HostNamePinId>(hostName);
					SetIfChanged<&HostInfo::IpAddress, &HostInfoPins::IpAddressPinId>(ip);
					if (uptime != -1)
						SetIfChanged<&HostInfo::UptimeS, &HostInfoPins::UptimeSecondsPinId>((uint32_t)uptime);
					std::unique_lock lock(Mutex);
					if (CV.wait_for(lock, std::chrono::seconds(1), [&stopToken] { return stopToken.stop_requested(); }))
						break;
				}
			});
		}
	}

	void RemoveListener(uuid nodeId)
	{
		Listeners.erase(nodeId);
		if (Listeners.empty())
		{
			nosEngine.LogD("Host: Stopping worker thread");
			Worker.request_stop();
			CV.notify_one();
			if (Worker.joinable())
				Worker.join();
		}
	}
	
	std::jthread Worker;
	std::mutex Mutex;
	std::condition_variable CV;
} GHostInfo{};

struct HostNode : NodeContext
{
	HostNode(nosFbNodePtr node) : NodeContext(node)
	{
		HostInfoPins pins {
			.HostNamePinId = *GetPinId(NOS_NAME("HostName")),
			.IpAddressPinId = *GetPinId(NOS_NAME("IpAddress")),
			.UptimeSecondsPinId = *GetPinId(NOS_NAME("UptimeSeconds"))
		};
		GHostInfo.AddListener(NodeId, std::move(pins));
	}

	~HostNode() override
	{
		GHostInfo.RemoveListener(NodeId);
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		return NOS_RESULT_SUCCESS;
	}
};

nosResult RegisterHost(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Host"), HostNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities