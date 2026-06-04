#include "viewer/RenderDocCapture.hpp"
#include "engine/Logging.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace te {

bool RenderDocCapture::init() {
    if (api_) return true;   // already initialized

#ifdef _WIN32
    // RenderDoc injects renderdoc.dll into the process when launched
    // via qrenderdoc.exe (or when the in-application API is used). If
    // the DLL is in our address space, we can get the API table.
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (!mod) {
        log_info("RenderDoc: renderdoc.dll not in process; capture API unavailable. "
                 "Launch via qrenderdoc.exe to enable capture.");
        return false;
    }

    auto get_api = reinterpret_cast<pRENDERDOC_GetAPI>(
        GetProcAddress(mod, "RENDERDOC_GetAPI"));
    if (!get_api) {
        log_warn("RenderDoc: renderdoc.dll loaded but RENDERDOC_GetAPI missing. "
                 "Incompatible RenderDoc version?");
        return false;
    }

    // API version 1.4.0 is the oldest version that includes
    // TriggerMultiFrameCapture. The 1.4.0 typedef is aliased to the
    // 1.7.0 struct in renderdoc_app.h, so we get all 1.4+ methods
    // (SetCaptureFileTemplate, SetCaptureTitle, TriggerMultiFrameCapture)
    // plus the older Start/EndFrameCapture path.
    if (get_api(eRENDERDOC_API_Version_1_4_0, reinterpret_cast<void**>(&api_)) != 1) {
        log_warn("RenderDoc: RENDERDOC_GetAPI(1_4_0) returned non-1. "
                 "Incompatible RenderDoc version?");
        api_ = nullptr;
        return false;
    }
    log_info("RenderDoc: capture API bound (1.4.0). "
             "Use qrenderdoc.exe to attach, or click 'Capture Frame' in the viewer.");
    return true;
#else
    log_info("RenderDoc: capture API not implemented for this platform.");
    return false;
#endif
}

void RenderDocCapture::trigger_capture(uint32_t num_frames) {
    if (!api_) return;
    // TriggerMultiFrameCapture is the artist-facing path. RenderDoc
    // pops up its own dialog so the user picks the save location.
    // We do NOT call Start/EndFrameCapture here -- RenderDoc handles
    // the frame boundaries itself based on its own present-tracking.
    api_->TriggerMultiFrameCapture(num_frames);
}

void RenderDocCapture::start_capture(RENDERDOC_DevicePointer device) {
    if (!api_) return;
    api_->StartFrameCapture(device, nullptr);
}

void RenderDocCapture::end_capture(RENDERDOC_DevicePointer device) {
    if (!api_) return;
    api_->EndFrameCapture(device, nullptr);
}

void RenderDocCapture::set_capture_title(const std::string& title) {
    if (!api_) return;
    api_->SetCaptureTitle(title.c_str());
}

} // namespace te
