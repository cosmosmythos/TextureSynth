#pragma once

#include <cstdint>
#include <functional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace te::glsl {

class GlslBuilder {
public:
    GlslBuilder() = default;

    void clear() {
        header_.clear();
        layouts_.clear();
        functions_.clear();
        main_.str({});
        main_.clear();
        main_started_ = false;
    }

    void add_header(std::string header) {
        if (!header.empty()) {
            header_ += std::move(header);
            header_ += '\n';
        }
    }

    void add_layout(std::string layout_decl) {
        layouts_.push_back(std::move(layout_decl));
    }

    void add_function(std::string func) {
        functions_.push_back(std::move(func));
    }

    void main_begin() {
        if (main_started_) {
            return;
        }

        main_started_ = true;
        main_ << "void main() {\n";
        main_ << "    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);\n";
        main_ << "    vec2 uv;\n";
        main_ << "    uv.x = float(coord.x) / float(max(pc.resolution_x, 1u));\n";
        main_ << "    uv.y = float(coord.y) / float(max(pc.resolution_y, 1u));\n";
        main_ << "    vec4 _result = vec4(0.0);\n\n";
    }

    void main_end(const std::string& output_expr) {
        main_begin();
        main_ << "\n    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, "
              << output_expr << ");\n";
        main_ << "}\n";
    }

    void main_end_multi(const std::vector<uint32_t>& slot_indices,
                        const std::function<std::string(uint32_t)>& var_for_slot) {
        main_begin();
        main_ << "\n";
        for (uint32_t i = 0; i < (uint32_t)slot_indices.size(); ++i) {
            main_ << "    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots["
                  << slot_indices[i] << "])], coord, " << var_for_slot(i) << ");\n";
        }
        main_ << "}\n";
    }

    void declare_local(const std::string& name) {
        main_begin();
        main_ << "    vec4 " << name << ";\n";
    }

    void declare_shared(uint32_t slot_count) {
        main_begin();
        main_ << "    shared vec4 spill_pool[" << slot_count << "];\n";
    }

    void spill_store(uint32_t slot, const std::string& reg_name) {
        main_begin();
        main_ << "    spill_pool[" << slot << "] = " << reg_name << ";\n";
    }

    std::string spill_load_expr(uint32_t slot) {
        return "spill_pool[" + std::to_string(slot) + "]";
    }

    void declare_external(const std::string& name, std::uint32_t slot) {
        main_begin();
        main_ << "    vec4 " << name
              << " = texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
              << slot << "])], coord, 0);\n";
    }

    void call_and_assign(
        const std::string& target,
        const std::string& func_name,
        const std::vector<std::string>& args)
    {
        main_begin();
        main_ << "    " << target << " = " << func_name << "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) {
                main_ << ", ";
            }
            main_ << args[i];
        }
        main_ << ");\n";
    }

    void call_void(
        const std::string& func_name,
        const std::vector<std::string>& args)
    {
        main_begin();
        main_ << "    " << func_name << "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) {
                main_ << ", ";
            }
            main_ << args[i];
        }
        main_ << ");\n";
    }

    void statement(const std::string& stmt) {
        main_begin();
        main_ << "    " << stmt << '\n';
    }

    void comment(const std::string& text) {
        main_begin();
        main_ << "\n    // " << text << '\n';
    }

    [[nodiscard]] std::string build() const {
        std::ostringstream ss;

        ss << header_;
        if (!header_.empty() && header_.back() != '\n') {
            ss << '\n';
        }
        ss << '\n';

        for (const auto& layout : layouts_) {
            ss << layout << '\n';
        }
        ss << '\n';

        for (const auto& function : functions_) {
            ss << function << "\n\n";
        }

        ss << main_.str();
        return ss.str();
    }

    std::ostream& main_stream() {
        main_begin();
        return main_;
    }

private:
    std::string header_;
    std::vector<std::string> layouts_;
    std::vector<std::string> functions_;
    std::ostringstream main_;
    bool main_started_ = false;
};

inline std::string compute_header(StorageFormat sf) {
    std::string storage_fmt = storage_format_glsl_qualifier(sf);
    return "#version 460\n"
R"(
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
)" + std::string("layout(set = 0, binding = 1, ") + storage_fmt + R"() writeonly uniform image2D u_storage[];
)" + R"(
layout(set = 0, binding = 2) uniform sampler samp_repeat;
layout(set = 0, binding = 3) uniform sampler samp_clamp;
layout(set = 0, binding = 4) uniform sampler samp_mirror;
layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } node_params[];
layout(constant_id = 0) const uint ts_pass_index = 0u;
layout(push_constant, std430) uniform PC {
    uint resolution_x, resolution_y, seed;
    float time;
    uint out_storage_slots[4];
    uint param_base_slot;
    uint input_count;
    uint param_ring_idx;
    uint in_sampled_slots[8];
    uint pass_index;
} pc;

struct TSTexture { int slot; vec2 inv_size; };

vec4 ts_sample(int local, vec2 uv) {
    return texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], samp_repeat), uv);
}
vec4 ts_sample_lod(int local, vec2 uv, float lod) {
    return textureLod(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], samp_repeat), uv, lod);
}
ivec2 ts_size(int local) {
    return textureSize(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], 0);
}
vec4  Sample      (TSTexture t, vec2 uv)            { return ts_sample(t.slot, uv); }
vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_sample_lod(t.slot, uv, lod); }
ivec2 GetSize     (TSTexture t)                     { return ts_size(t.slot); }
vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }
)";
}

// Deprecated shim: preserves old behavior (F32 qualifier regardless of depth).
// New callers should pass StorageFormat explicitly so shader and image formats match.
inline std::string compute_header(ChannelFormat fmt = ChannelFormat::RGBA) {
    return compute_header(StorageFormat{fmt, BitDepth::F32});
}

inline std::string format_helpers() {
    return R"(
vec4 _fmt_mono(vec4 v) { return vec4(v.x, 0.0, 0.0, 1.0); }
vec4 _fmt_uv(vec4 v) { return vec4(v.xy, 0.0, 1.0); }
vec4 _fmt_rgb(vec4 v) { return vec4(v.rgb, 1.0); }
vec4 _fmt_rgba(vec4 v) { return v; }
)";
}

} // namespace te::glsl
