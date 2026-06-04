#pragma once
#include "third_party/renderdoc_app.h"
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>

namespace te {

// Thin runtime loader for the RenderDoc in-application capture API.
//
// RenderDoc injects its `renderdoc.dll` into a process when launched via
// qrenderdoc.exe's "Inject into Process" (or via the in-app API). Once
// injected, the DLL lives in this process's address space and exports
// `RENDERDOC_GetAPI` which hands us a function table for starting,
// ending, and configuring captures.
//
// We do NOT link to renderdoc.lib at build time. The DLL is found at
// runtime via GetModuleHandleA, so the viewer binary has zero build
// dependency on RenderDoc. If RenderDoc is not attached, every method
// is a no-op -- the viewer runs exactly as it does today.
//
// Industry standard pattern -- see:
//   https://renderdoc.org/docs/in_application_api.html
//   Unreal Engine: Source/Runtime/RenderCore/Private/RenderDoc*.cpp
//   Godot:         modules/renderdoc/renderdoc.cpp
class RenderDocCapture {
public:
    // Look up the injected renderdoc.dll in this process. Returns true
    // if the API was successfully obtained. Safe to call multiple times.
    bool init();

    // Did init() find and bind the API?
    bool is_available() const { return api_ != nullptr; }

    // Trigger N frames of capture via RenderDoc's own trigger. This is
    // the path that lets the artist pick the save location via the
    // RenderDoc UI (Save/Discard dialog). N=1 captures the next frame.
    // No-op if is_available() is false.
    void trigger_capture(uint32_t num_frames = 1);

    // Manual capture around a single frame. Caller is responsible for
    // calling end_capture() on the same device pointer after the work
    // has been submitted. The device pointer is the dispatchable
    // handle -- for Vulkan, use RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE
    // or RENDERDOC_DEVICEPOINTER_FROM_VKDEVICE.
    void start_capture(RENDERDOC_DevicePointer device);
    void end_capture(RENDERDOC_DevicePointer device);

    // Convenience: set the title in RenderDoc's capture log.
    void set_capture_title(const std::string& title);

    // Convenience: Vulkan-specific helpers. The instance pointer is
    // the actual VkInstance cast to void*; renderdoc treats it as an
    // opaque dispatchable handle.
    void start_capture_vk(VkInstance instance) {
        start_capture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance));
    }
    void end_capture_vk(VkInstance instance) {
        end_capture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance));
    }

private:
    RENDERDOC_API_1_4_0* api_ = nullptr;
};

} // namespace te
