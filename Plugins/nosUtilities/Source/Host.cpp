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
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <netdb.h>
#include <ctime>
#include <unistd.h>
#endif

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
	std::string ip;

#if defined(_WIN32)
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
	char hostbuffer[256];
	char* IPbuffer;
	struct hostent* host_entry;
	int hostname;
	// To retrieve hostname
	hostname = gethostname(hostbuffer, sizeof(hostbuffer));
	if (hostname == -1) {
		perror("gethostname");
		return "";
	}
	// To retrieve host information
	host_entry = gethostbyname(hostbuffer);
	if (host_entry == NULL) {
		perror("gethostbyname");
		return "";
	}
	// To convert an Internet network address into ASCII string
	IPbuffer = inet_ntoa(*((struct in_addr*)host_entry->h_addr_list[0]));
	ip = IPbuffer;
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
	if (clock_gettime(CLOCK_UPTIME, &ts) == 0) {
		return ts.tv_sec;
	}
	else {
		return -1;
	}
#else
	return -1;  // Unsupported platform
#endif
}


struct HostNode : NodeContext
{
	struct HostInfo
	{
		std::pair<nosUUID, std::string> HostName;
		std::pair<nosUUID, std::string> IpAddress;
		std::pair<nosUUID, uint32_t> UptimeS;

		template <auto Member, typename T>
		void Set(T const& value)
		{
			auto& member = this->*Member;
			auto& old = member.second;
			if (old != value)
			{
				old = value;
				if constexpr (std::is_same_v<T, std::string>)
					nosEngine.SetPinValue(member.first, nos::Buffer(value.c_str(), value.size() + 1));
				else
				{
					auto size = sizeof(value);
					nosEngine.SetPinValue(member.first, nos::Buffer(reinterpret_cast<const char*>(&value), size));
				}
			}
		}
	};

	HostNode(const nosFbNode* node) : NodeContext(node)
	{
		// TODO: Use OS callbacks to subscribe to changes.
		BackgroundThread = std::thread([this,
			hostNamePinId = *GetPinId(NOS_NAME("HostName")),
			ipAddressPinId = *GetPinId(NOS_NAME("IpAddress")),
			uptimePinId = *GetPinId(NOS_NAME("UptimeSeconds"))
		]()
			{
				HostInfo info{
					.HostName = {hostNamePinId, ""},
					.IpAddress = {ipAddressPinId, ""},
					.UptimeS = {uptimePinId, 0}
				};
				while (true)
				{
					auto hostName = GetHostName();
					auto ip = GetIpv4Address();
					auto uptime = GetUpTime();
					info.Set<&HostInfo::HostName>(hostName);
					info.Set<&HostInfo::IpAddress>(ip);
					if (uptime != -1)
						info.Set<&HostInfo::UptimeS>((uint32_t)uptime);
					std::unique_lock lock(Mutex);
					if (CV.wait_for(lock, std::chrono::seconds(1), [this] { return ShouldExit.load(); }))
						break;
				}
			});
	}

	~HostNode()
	{
		ShouldExit = true;
		CV.notify_all();
		if (BackgroundThread.joinable())
			BackgroundThread.join();
	}

	nosResult ExecuteNode(nosNodeExecuteParams* params) override
	{
		return NOS_RESULT_SUCCESS;
	}

protected:
	std::thread BackgroundThread;
	std::condition_variable CV;
	std::mutex Mutex;
	std::atomic_bool ShouldExit = false;
};

nosResult RegisterHost(nosNodeFunctions* fn)
{
	NOS_BIND_NODE_CLASS(NOS_NAME("Host"), HostNode, fn);
	return NOS_RESULT_SUCCESS;
}

} // namespace nos::utilities