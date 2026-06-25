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
constexpr uint32_t MAX_PASS_OUTPUTS = 4;

struct PassPushConstants {
    PushConstants global;
    uint32_t      out_storage_slots[MAX_PASS_OUTPUTS];
    uint32_t      param_base_slot;
    uint32_t      input_count;
    uint32_t      param_ring_idx;
    uint32_t      in_sampled_slots[MAX_PASS_INPUTS];
    uint32_t      pass_index;  // sub-pass index within multi-pass chain (0, 1, ...)
};
static_assert(sizeof(PassPushConstants) == 80, "PassPushConstants size mismatch");

} // namespace te