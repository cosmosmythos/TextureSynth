#include <gtest/gtest.h>
#include "engine/Graph.hpp"

using namespace te;

// Verify the channel x depth cross-product resolves to the expected VkFormat.
TEST(StorageFormat, VkFormatMatrix) {
    // Mono
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Mono, BitDepth::F8}),  VK_FORMAT_R8_UNORM);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Mono, BitDepth::F16}), VK_FORMAT_R16_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Mono, BitDepth::F32}), VK_FORMAT_R32_SFLOAT);
    // UV
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::UV, BitDepth::F8}),  VK_FORMAT_R8G8_UNORM);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::UV, BitDepth::F16}), VK_FORMAT_R16G16_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::UV, BitDepth::F32}), VK_FORMAT_R32G32_SFLOAT);
    // RGB (packed to RGBA)
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGB, BitDepth::F8}),  VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGB, BitDepth::F16}), VK_FORMAT_R16G16B16A16_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGB, BitDepth::F32}), VK_FORMAT_R32G32B32A32_SFLOAT);
    // RGBA
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGBA, BitDepth::F8}),  VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGBA, BitDepth::F16}), VK_FORMAT_R16G16B16A16_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::RGBA, BitDepth::F32}), VK_FORMAT_R32G32B32A32_SFLOAT);
}

// ID and Metadata are special -- ignore depth.
TEST(StorageFormat, SpecialFormatsIgnoreDepth) {
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::ID, BitDepth::F8}),  VK_FORMAT_R32_UINT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::ID, BitDepth::F16}), VK_FORMAT_R32_UINT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::ID, BitDepth::F32}), VK_FORMAT_R32_UINT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Metadata, BitDepth::F8}),  VK_FORMAT_R32G32B32A32_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Metadata, BitDepth::F16}), VK_FORMAT_R32G32B32A32_SFLOAT);
    EXPECT_EQ(storage_format_to_vk({ChannelFormat::Metadata, BitDepth::F32}), VK_FORMAT_R32G32B32A32_SFLOAT);
}

// Bytes per pixel drives the VRAM budget check.
TEST(StorageFormat, BytesPerPixel) {
    EXPECT_EQ(storage_format_bytes({ChannelFormat::Mono, BitDepth::F8}),  1u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::Mono, BitDepth::F16}), 2u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::Mono, BitDepth::F32}), 4u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::UV,   BitDepth::F16}), 4u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::RGBA, BitDepth::F8}),  4u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::RGBA, BitDepth::F16}), 8u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::RGBA, BitDepth::F32}), 16u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::ID,   BitDepth::F32}), 4u);
    EXPECT_EQ(storage_format_bytes({ChannelFormat::Metadata, BitDepth::F32}), 16u);
}

// GLSL qualifier MUST match VkFormat exactly -- this is the root-cause fix
// for the old bug where shader declared "r32f" but image was R16_SFLOAT.
// Note: GLSL uses 'f' suffix (rgba16f), not VkFormat's _SFLOAT spelling.
TEST(StorageFormat, GlslQualifierMatchesVkFormat) {
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::Mono, BitDepth::F16}), "r16f");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::Mono, BitDepth::F32}), "r32f");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::UV,   BitDepth::F16}), "rg16f");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::RGBA, BitDepth::F8}),  "rgba8");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::RGBA, BitDepth::F16}), "rgba16f");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::RGBA, BitDepth::F32}), "rgba32f");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::ID,   BitDepth::F32}), "r32ui");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::Mono, BitDepth::F8}), "r8");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::UV,   BitDepth::F8}), "rg8");
    EXPECT_EQ(storage_format_glsl_qualifier({ChannelFormat::RGB,  BitDepth::F16}), "rgba16f");
}

// Deprecated shim still works for migration period.
TEST(StorageFormat, DeprecatedShimMatchesF16) {
    EXPECT_EQ(channel_to_vk_format(ChannelFormat::Mono), VK_FORMAT_R16_SFLOAT);
    EXPECT_EQ(channel_to_vk_format(ChannelFormat::UV),   VK_FORMAT_R16G16_SFLOAT);
    EXPECT_EQ(channel_to_vk_format(ChannelFormat::ID),   VK_FORMAT_R32_UINT);
}

TEST(StorageFormat, Equality) {
    StorageFormat a{ChannelFormat::Mono, BitDepth::F32};
    StorageFormat b{ChannelFormat::Mono, BitDepth::F32};
    StorageFormat c{ChannelFormat::Mono, BitDepth::F16};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}
