#ifndef CUDA_SUBSYS_COMMON_H_INCLUDED
#define CUDA_SUBSYS_COMMON_H_INCLUDED

namespace Descriptor {


#include <Windows.h>
#include <sddl.h>
#include <ntdef.h>

struct SecurityDescriptor {
        static void* GetDefaultSecurityDescriptor()
        {
            #if defined(__linux__)
            return;
            #elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
            static OBJECT_ATTRIBUTES objAttributes;
            static bool objAttributesConfigured = false;

            if (!objAttributesConfigured) {
                PSECURITY_DESCRIPTOR secDesc;
                BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1, &secDesc, NULL);
                InitializeObjectAttributes(
                    &objAttributes,
                    NULL,
                    0,
                    NULL,
                    secDesc
                );

                objAttributesConfigured = true;
            }

            return &objAttributes;
            #endif
        }
};
}

#endif //CUDA_SUBSYS_COMMON_H_INCLUDED
