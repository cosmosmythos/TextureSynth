#pragma once
#include <cstdint>

namespace te {

// Global frame data (matches old layout, first 16 bytes).
struct PushConstants {
    uint32_t resolution_x;
    uint32_t resolution_y;
    uint32_t seed;
    float    time;
};
static_assert(sizeof(PushConstants) == 16, "PushConstants size mismatch");

constexpr uint32_t MAX_PASS_INPUTS = 8;

struct PassPushConstants {
    PushConstants global;                  // bytes 0..15
    uint32_t      out_storage_slot;        // 16
    uint32_t      param_base_slot;         // 20 (SSBO float index)
    uint32_t      input_count;             // 24
    uint32_t      param_ring_idx;          // 28 
    uint32_t      in_sampled_slots[MAX_PASS_INPUTS]; // 32..63
};
static_assert(sizeof(PassPushConstants) == 64, "PassPushConstants must be 64 bytes");

} // namespace te