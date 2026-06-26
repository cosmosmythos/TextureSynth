#include "engine/Graph.hpp"

#include <string>

namespace te {

namespace {

constexpr StorageFormatInfo kStorageFormats[] = {
    {{ChannelFormat::Mono, BitDepth::F8},  VK_FORMAT_R8_UNORM,              "R8_UNORM",              "r8",      1, 1, 1},
    {{ChannelFormat::Mono, BitDepth::F16}, VK_FORMAT_R16_SFLOAT,            "R16_SFLOAT",            "r16f",    2, 1, 1},
    {{ChannelFormat::Mono, BitDepth::F32}, VK_FORMAT_R32_SFLOAT,            "R32_SFLOAT",            "r32f",    4, 1, 1},

    {{ChannelFormat::UV,   BitDepth::F8},  VK_FORMAT_R8G8_UNORM,            "R8G8_UNORM",            "rg8",     2, 2, 2},
    {{ChannelFormat::UV,   BitDepth::F16}, VK_FORMAT_R16G16_SFLOAT,         "R16G16_SFLOAT",         "rg16f",   4, 2, 2},
    {{ChannelFormat::UV,   BitDepth::F32}, VK_FORMAT_R32G32_SFLOAT,         "R32G32_SFLOAT",         "rg32f",   8, 2, 2},

    {{ChannelFormat::RGB,  BitDepth::F8},  VK_FORMAT_R8G8B8A8_UNORM,        "R8G8B8A8_UNORM",        "rgba8",   4, 3, 4},
    {{ChannelFormat::RGB,  BitDepth::F16}, VK_FORMAT_R16G16B16A16_SFLOAT,   "R16G16B16A16_SFLOAT",   "rgba16f", 8, 3, 4},
    {{ChannelFormat::RGB,  BitDepth::F32}, VK_FORMAT_R32G32B32A32_SFLOAT,   "R32G32B32A32_SFLOAT",   "rgba32f", 16, 3, 4},

    {{ChannelFormat::RGBA, BitDepth::F8},  VK_FORMAT_R8G8B8A8_UNORM,        "R8G8B8A8_UNORM",        "rgba8",   4, 4, 4},
    {{ChannelFormat::RGBA, BitDepth::F16}, VK_FORMAT_R16G16B16A16_SFLOAT,   "R16G16B16A16_SFLOAT",   "rgba16f", 8, 4, 4},
    {{ChannelFormat::RGBA, BitDepth::F32}, VK_FORMAT_R32G32B32A32_SFLOAT,   "R32G32B32A32_SFLOAT",   "rgba32f", 16, 4, 4},
};

constexpr size_t kStorageFormatCount = sizeof(kStorageFormats) / sizeof(kStorageFormats[0]);

} // namespace

const StorageFormatInfo& storage_format_info(StorageFormat fmt) {
    for (const auto& info : kStorageFormats) {
        if (info.storage == fmt) return info;
    }
    return storage_format_info(StorageFormat{ChannelFormat::RGBA, BitDepth::F16});
}

const StorageFormatInfo* storage_format_info_table(size_t& count) {
    count = kStorageFormatCount;
    return kStorageFormats;
}

VkFormat storage_format_to_vk(StorageFormat fmt) {
    return storage_format_info(fmt).vk_format;
}

uint32_t storage_format_bytes(StorageFormat fmt) {
    return storage_format_info(fmt).bytes_per_pixel;
}

std::string storage_format_glsl_qualifier(StorageFormat fmt) {
    return storage_format_info(fmt).glsl_qualifier;
}

bool storage_format_has_exact_vk_channels(StorageFormat fmt) {
    const auto& info = storage_format_info(fmt);
    return info.logical_channels == info.stored_channels;
}

const char* vk_format_name(VkFormat fmt) {
    for (const auto& info : kStorageFormats) {
        if (info.vk_format == fmt) return info.vk_name;
    }
    switch (fmt) {
        case VK_FORMAT_R32_UINT: return "R32_UINT";
        case VK_FORMAT_R16_UINT: return "R16_UINT";
        case VK_FORMAT_R8_UINT:  return "R8_UINT";
        default:                 return "UNKNOWN_FORMAT";
    }
}

uint32_t vk_format_bytes(VkFormat fmt) {
    for (const auto& info : kStorageFormats) {
        if (info.vk_format == fmt) return info.bytes_per_pixel;
    }
    switch (fmt) {
        case VK_FORMAT_R32_UINT: return 4;
        case VK_FORMAT_R16_UINT: return 2;
        case VK_FORMAT_R8_UINT:  return 1;
        default:                 return 0;
    }
}

} // namespace te
