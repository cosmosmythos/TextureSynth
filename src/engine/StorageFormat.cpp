#include "engine/Graph.hpp"
#include <string>

namespace te {

// Lookup table for the channel x depth cross-product.
// Indexing: [ChannelFormat][BitDepth]. Special cases (ID, Metadata) ignore depth.
struct VkFormatEntry { VkFormat f8, f16, f32; };
static constexpr VkFormatEntry kFormatTable[] = {
    // Mono
    { VK_FORMAT_R8_UNORM,           VK_FORMAT_R16_SFLOAT,           VK_FORMAT_R32_SFLOAT },
    // UV
    { VK_FORMAT_R8G8_UNORM,         VK_FORMAT_R16G16_SFLOAT,         VK_FORMAT_R32G32_SFLOAT },
    // RGB (packed into RGBA; alpha forced to 1.0)
    { VK_FORMAT_R8G8B8A8_UNORM,     VK_FORMAT_R16G16B16A16_SFLOAT,   VK_FORMAT_R32G32B32A32_SFLOAT },
    // RGBA
    { VK_FORMAT_R8G8B8A8_UNORM,     VK_FORMAT_R16G16B16A16_SFLOAT,   VK_FORMAT_R32G32B32A32_SFLOAT },
};

VkFormat storage_format_to_vk(StorageFormat fmt) {
    switch (fmt.channels) {
        case ChannelFormat::ID:       return VK_FORMAT_R32_UINT;
        case ChannelFormat::Metadata: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: break;
    }
    const auto idx = static_cast<size_t>(fmt.channels);
    if (idx >= std::size(kFormatTable))
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkFormatEntry& e = kFormatTable[idx];
    switch (fmt.depth) {
        case BitDepth::F8:  return e.f8;
        case BitDepth::F16: return e.f16;
        case BitDepth::F32: return e.f32;
    }
    return VK_FORMAT_R16G16B16A16_SFLOAT;
}

uint32_t storage_format_bytes(StorageFormat fmt) {
    // Special cases first.
    switch (fmt.channels) {
        case ChannelFormat::ID:       return 4;  // R32_UINT
        case ChannelFormat::Metadata: return 16; // RGBA32F
        default: break;
    }
    const auto idx = static_cast<size_t>(fmt.channels);
    if (idx >= std::size(kFormatTable))
        return 8;
    const VkFormatEntry& e = kFormatTable[idx];
    switch (fmt.depth) {
        case BitDepth::F8: {
            switch (fmt.channels) {
                case ChannelFormat::Mono: return 1;
                case ChannelFormat::UV:   return 2;
                default:                  return 4;  // RGB/RGBA packed
            }
        }
        case BitDepth::F16: {
            switch (fmt.channels) {
                case ChannelFormat::Mono: return 2;
                case ChannelFormat::UV:   return 4;
                default:                  return 8;  // RGB/RGBA
            }
        }
        case BitDepth::F32: {
            switch (fmt.channels) {
                case ChannelFormat::Mono: return 4;
                case ChannelFormat::UV:   return 8;
                default:                  return 16; // RGB/RGBA
            }
        }
    }
    (void)e;
    return 8;
}

std::string storage_format_glsl_qualifier(StorageFormat fmt) {
    // GLSL image2D format qualifiers — NOT the same spelling as VkFormat names.
    // VkFormat uses _SFLOAT suffix (R16G16B16A16_SFLOAT); GLSL uses just 'f'
    // (rgba16f). The mapping below must produce the Vulkan-spec image format
    // qualifier matching storage_format_to_vk (validation layer enforces this).
    switch (fmt.channels) {
        case ChannelFormat::ID:       return "r32ui";
        case ChannelFormat::Metadata: return "rgba32f";
        default: break;
    }
    switch (fmt.depth) {
        case BitDepth::F8:
            switch (fmt.channels) {
                case ChannelFormat::Mono: return "r8";
                case ChannelFormat::UV:   return "rg8";
                default:                  return "rgba8";
            }
        case BitDepth::F16:
            switch (fmt.channels) {
                case ChannelFormat::Mono: return "r16f";
                case ChannelFormat::UV:   return "rg16f";
                default:                  return "rgba16f";
            }
        case BitDepth::F32:
            switch (fmt.channels) {
                case ChannelFormat::Mono: return "r32f";
                case ChannelFormat::UV:   return "rg32f";
                default:                  return "rgba32f";
            }
    }
    return "rgba16f";
}

} // namespace te
