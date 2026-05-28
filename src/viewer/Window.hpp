#pragma once
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>

namespace te {

class Window {
public:
    bool init(uint32_t width, uint32_t height, const std::string& title);
    void shutdown();

    bool should_close() const;
    void poll_events();

    GLFWwindow* handle() const { return window_; }
    VkSurfaceKHR create_surface(VkInstance instance);
    
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    
    bool was_resized() const { return resized_; }
    void reset_resized_flag() { resized_ = false; }
    void wait_events_if_minimized();

private:
    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);

    GLFWwindow* window_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool resized_ = false;
};

} // namespace te
