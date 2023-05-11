#include "AJADevice.h"

#include <windows.h>
#include <winnt.h>


std::string GetLastErrorAsString()
{
    // Get the error message ID, if any.
    DWORD errorMessageID = ::GetLastError();
    if (errorMessageID == 0)
    {
        return std::string(); // No error message has been recorded
    }

    LPSTR messageBuffer = nullptr;

    // Ask Win32 to give us the string version of that message ID.
    // The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL,
                                 errorMessageID,
                                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                 (LPSTR)&messageBuffer,
                                 0,
                                 NULL);

    // Copy the error message into a std::string.
    std::string message(messageBuffer, size);

    // Free the Win32's string's buffer.
    LocalFree(messageBuffer);

    return message;
}

void* AJADevice::Alloc(size_t size, bool read)
{
    // MEM_COMMIT | MEM_RESERVE;
    // MEM_RESET | MEM_RESET_UNDO;
    // MEM_LARGE_PAGES | MEM_PHYSICAL | MEM_TOP_DOWN;

    // MEM_PHYSICAL -> { MEM_RESERVE };

    // PAGE_GUARD x { PAGE_NOACCESS };
    // PAGE_NOCACHE x { PAGE_GUARD, PAGE_NOACCESS, PAGE_WRITECOMBINE };
    // PAGE_WRITECOMBINE x { PAGE_NOACCESS, PAGE_GUARD, PAGE_NOCACHE };

    if(size & 4095) size += 4096 - (size & 4095);
    // auto buf = VirtualAllocEx(GetCurrentProcess(), 0, size, MEM_COMMIT, PAGE_NOACCESS);
    auto buf = GlobalAlloc(GMEM_FIXED, size);
    if(!buf)
    {
        std::cout << GetLastErrorAsString() << "\n--0--\n";
        return 0;
    }
    auto ptr = GlobalLock(buf);
    if(!ptr)
    {
        std::cout << GetLastErrorAsString() << "\n--1--\n";
        return 0;
    }
    return ptr;
}

void AJADevice::Dealloc(void* ptr, size_t size)
{
    if(size & 4095) size += 4096 - (size & 4095);
    VirtualFreeEx(GetCurrentProcess(), ptr, size, MEM_RELEASE);
}