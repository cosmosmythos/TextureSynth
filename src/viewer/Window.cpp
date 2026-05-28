#include "viewer/Window.hpp"
#include <stdexcept>
#include <iostream>

namespace te {

bool Window::init(uint32_t width, uint32_t height, const std::string& title) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    
    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window\n";
        return false;
    }

    width_ = width;
    height_ = height;

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebuffer_resize_callback);

    return true;
}

void Window::shutdown() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

bool Window::should_close() const {
    return glfwWindowShouldClose(window_);
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::wait_events_if_minimized() {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window_, &w, &h);
        glfwWaitEvents();
        if (glfwWindowShouldClose(window_)) break;
    }
}

VkSurfaceKHR Window::create_surface(VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, window_, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    return surface;
}

void Window::framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    app->width_ = width;
    app->height_ = height;
    app->resized_ = true;
}

} // namespace te
