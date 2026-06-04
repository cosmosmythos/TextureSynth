// Tests for Stage 2: ShaderVariantKey + feature_flags + specialization plumbing.
// Also covers Stage 2 emission: format_override → TS_FORMAT #define + format
// post-process in emit_node_shader.
//
// Three layers of coverage:
//   (1) Direct unit tests on ShaderVariantKey — no engine/Vulkan needed.
//   (2) Integration tests through GraphCompiler::compile — verifies the
//       variant key is actually populated from inst->format_override.
//   (3) GLSL emission tests — verifies the format post-process actually
//       appears in the emitted shader source, producing per-format .spv variants.
//
// These tests existed nowhere before Stage 2 — the entire feature_flags /
// specialization / emission pathway was uncovered.

#include <gtest/gtest.h>
#include <set>
#include <string>
#include "engine/ShaderVariantKey.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/ComputePipeline.hpp"

using namespace te;

namespace {

// ── Helpers for GraphCompiler integration tests ─────────────────────────
NodeType make_type(const std::string& id, uint32_t n_inputs, uint32_t n_outputs) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    for (uint32_t i = 0; i < n_inputs; ++i) {
        Socket s; s.name = "in" + std::to_string(i); s.type = SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_outputs; ++i) {
        Socket s; s.name = "out" + std::to_string(i); s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    // Non-empty glsl_function makes the compiler treat this as a Dispatch pass
    // (not a ResourceBind), so build_variant_key is called. The body is
    // arbitrary — these tests only inspect the variant_key, not the GLSL.
    t.glsl_function = "vec4 node_" + id + "(vec2 uv) { return vec4(0.0); }";
    return t;
}

NodeLibrary make_library() {
    NodeLibrary lib;
    lib.add_public(make_type("source",      0, 1));
    lib.add_public(make_type("passthrough", 1, 1));
    return lib;
}

} // namespace

// ════════════════════════════════════════════════════════════════════════
// Layer 1: Direct ShaderVariantKey unit tests
// ════════════════════════════════════════════════════════════════════════

TEST(ShaderVariantKey, DefaultKeyHasAllFieldsZero) {
    ShaderVariantKey k;
    EXPECT_EQ(k.node_type_id, "");
    EXPECT_EQ(k.input_count,       0u);
    EXPECT_EQ(k.param_socket_mask, 0u);
    EXPECT_EQ(k.feature_flags,     0u);
    EXPECT_EQ(k.specialization_count, 0u);
    for (auto v : k.specialization) EXPECT_EQ(v, 0u);
}

TEST(ShaderVariantKey, HashIsDeterministic) {
    ShaderVariantKey a;
    a.node_type_id = "perlin";
    a.input_count  = 0;
    a.feature_flags = 0u;  // Mono
    ShaderVariantKey b = a;
    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(a, b);
}

TEST(ShaderVariantKey, HashChangesWithFeatureFlags) {
    // Same node, different feature_flags → different hash.
    // This is the core invariant: format_override is in the cache key.
    ShaderVariantKey mono;
    mono.node_type_id = "perlin";
    mono.feature_flags = 0u;  // Mono

    ShaderVariantKey uv;
    uv.node_type_id = "perlin";
    uv.feature_flags = 1u;    // UV

    EXPECT_NE(mono.hash(), uv.hash());
    EXPECT_NE(mono, uv);
}

TEST(ShaderVariantKey, HashChangesWithSpecialization) {
    ShaderVariantKey a;
    a.node_type_id = "x";
    a.specialization_count = 1;
    a.specialization[0] = 7;

    ShaderVariantKey b;
    b.node_type_id = "x";
    b.specialization_count = 1;
    b.specialization[0] = 9;

    EXPECT_NE(a.hash(), b.hash());
    EXPECT_NE(a, b);
}

TEST(ShaderVariantKey, SpecializationTailDoesNotAffectHashWhenCountIsZero) {
    // Garbage in specialization[5] is irrelevant if count == 0. This
    // matters because operator==/hash() only walks up to specialization_count.
    ShaderVariantKey a;
    a.node_type_id = "x";
    a.specialization_count = 0;
    a.specialization[5] = 0xDEADBEEF;

    ShaderVariantKey b;
    b.node_type_id = "x";
    b.specialization_count = 0;
    b.specialization[5] = 0xCAFEBABE;

    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(a, b);
}

TEST(ShaderVariantKey, EqualityIsReflexiveSymmetricTransitive) {
    ShaderVariantKey base;
    base.node_type_id = "perlin";
    base.input_count  = 0;
    base.feature_flags = 2u;  // RGB
    base.specialization_count = 2;
    base.specialization[0] = 1;
    base.specialization[1] = 2;

    ShaderVariantKey copy = base;
    EXPECT_EQ(base, copy);
    EXPECT_EQ(base.hash(), copy.hash());

    // Mutate, check inequality.
    copy.specialization[1] = 3;
    EXPECT_NE(base, copy);
}

TEST(ShaderVariantKey, StdHashIsConsistentWithOperatorEq) {
    // If a == b, then std::hash(a) == std::hash(b). Required by unordered_map.
    std::hash<ShaderVariantKey> hasher;
    ShaderVariantKey a;
    a.node_type_id = "perlin";
    a.specialization_count = 1;
    a.specialization[0] = 42;
    ShaderVariantKey b = a;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST(ShaderVariantKey, EpochIsMixedIn) {
    // Sanity: even a fully-default key has a non-zero hash because epoch
    // is mixed. The exact value is not the contract; just that it differs
    // from "what we'd get without epoch".
    ShaderVariantKey k;
    EXPECT_NE(k.hash(), 1469598103934665603ull);  // FNV offset basis
}

TEST(ShaderVariantKey, EightDistinctSpecializationValuesProduceEightDistinctHashes) {
    // All-ones-of-each-uint32: every index distinct, every value distinct.
    std::set<uint64_t> hashes;
    for (uint32_t i = 0; i < 8; ++i) {
        ShaderVariantKey k;
        k.node_type_id = "x";
        k.specialization_count = static_cast<uint32_t>(i) + 1;
        k.specialization[i]    = 0xC0FFEEu + i;
        hashes.insert(k.hash());
    }
    EXPECT_EQ(hashes.size(), 8u);
}

// ════════════════════════════════════════════════════════════════════════
// Layer 2: GraphCompiler integration — format_override → feature_flags
// ════════════════════════════════════════════════════════════════════════

TEST(GraphCompilerVariantKey, FormatOverrideFlowsIntoFeatureFlags) {
    // Two graphs, same node type, different format_override.
    // The cache key (and therefore the resulting feature_flags) must differ.
    auto lib = make_library();

    auto build = [&](ChannelFormat fmt) {
        Graph g;
        NodeInstance inst;
        inst.id = 1;
        inst.type_id = "source";
        inst.format_override = fmt;
        g.nodes.push_back(inst);
        g.output_node = 1;
        return g;
    };

    Graph g_mono = build(ChannelFormat::Mono);
    Graph g_uv   = build(ChannelFormat::UV);

    auto r_mono = validate_graph(g_mono, lib);
    auto r_uv   = validate_graph(g_uv,   lib);
    ASSERT_TRUE(r_mono.success) << r_mono.error;
    ASSERT_TRUE(r_uv.success)   << r_uv.error;

    auto c_mono = GraphCompiler::compile(r_mono.ir, lib);
    auto c_uv   = GraphCompiler::compile(r_uv.ir,   lib);
    ASSERT_TRUE(c_mono.success) << c_mono.error;
    ASSERT_TRUE(c_uv.success)   << c_uv.error;

    ASSERT_EQ(c_mono.pass_plan.passes.size(), 1u);
    ASSERT_EQ(c_uv.pass_plan.passes.size(),   1u);

    const auto& k_mono = c_mono.pass_plan.passes[0].variant_key;
    const auto& k_uv   = c_uv.pass_plan.passes[0].variant_key;

    // ChannelFormat::Mono = 0, ChannelFormat::UV = 1.
    EXPECT_EQ(k_mono.feature_flags & 0x7u, 0u);
    EXPECT_EQ(k_uv.feature_flags   & 0x7u, 1u);
    EXPECT_NE(k_mono.hash(), k_uv.hash());
    EXPECT_NE(k_mono, k_uv);
}

TEST(GraphCompilerVariantKey, SameFormatProducesSameKey) {
    // Two graphs with the same format_override must produce the same key.
    auto lib = make_library();

    auto build = [&](NodeId id) {
        Graph g;
        NodeInstance inst;
        inst.id = id;
        inst.type_id = "source";
        inst.format_override = ChannelFormat::RGBA;
        g.nodes.push_back(inst);
        g.output_node = id;
        return g;
    };

    auto r1 = validate_graph(build(1), lib);
    auto r2 = validate_graph(build(2), lib);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);

    auto c1 = GraphCompiler::compile(r1.ir, lib);
    auto c2 = GraphCompiler::compile(r2.ir, lib);
    ASSERT_TRUE(c1.success);
    ASSERT_TRUE(c2.success);

    EXPECT_EQ(c1.pass_plan.passes[0].variant_key.hash(),
              c2.pass_plan.passes[0].variant_key.hash());
    // ChannelFormat::RGBA = 3.
    EXPECT_EQ(c1.pass_plan.passes[0].variant_key.feature_flags & 0x7u, 3u);
}

TEST(GraphCompilerVariantKey, AllSixChannelFormatsAreDistinctKeys) {
    // Every ChannelFormat enum value (0..5) must produce a distinct
    // feature_flags low 3 bits — no collision in the low 3 bits.
    auto lib = make_library();
    const ChannelFormat fmts[] = {
        ChannelFormat::Mono, ChannelFormat::UV, ChannelFormat::RGB,
        ChannelFormat::RGBA, ChannelFormat::ID,  ChannelFormat::Metadata,
    };
    std::set<uint32_t> low_bits;
    std::set<uint64_t> hashes;
    for (auto fmt : fmts) {
        Graph g;
        NodeInstance inst; inst.id = 1; inst.type_id = "source";
        inst.format_override = fmt;
        g.nodes.push_back(inst);
        g.output_node = 1;
        auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
        auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);
        const auto& k = c.pass_plan.passes[0].variant_key;
        low_bits.insert(k.feature_flags & 0x7u);
        hashes.insert(k.hash());
    }
    EXPECT_EQ(low_bits.size(), 6u);  // 6 distinct low-3-bit patterns
    EXPECT_EQ(hashes.size(),   6u);  // 6 distinct hashes
}

// Note: a ComputePipeline::create-with-nullptr-spec test was considered but
// removed — it would require either a real SPIR-V (a much heavier setup
// involving the shader compiler) or an empty SPIR-V (which the Vulkan
// validation layer rejects with an SEH exception). The signature shape is
// already locked by the compiler (any future refactor that breaks the
// default-arg would fail to build the existing Engine.cpp call sites).
// End-to-end pipeline creation with nullptr spec is covered by
// Engine.EndToEndSingleNodeRender, which exercises the full install path
// with the build_spec_info helper returning nullptr for spec_count == 0.

// ════════════════════════════════════════════════════════════════════════
// Layer 3: emit_node_shader side — format_override → TS_FORMAT + post-process
// ════════════════════════════════════════════════════════════════════════
// These tests inspect the GLSL *source* emitted by GraphCompiler::compile,
// not the compiled SPIR-V. Pure C++ — no Vulkan, no shaderc needed.
// Proves the format actually appears in the .spv input, not just the cache key.

namespace {

// Noise-style node: returns the canonical vec4(noise, grad.x, grad.y, 1)
// that the format post-process knows how to fold.
NodeType make_noise_type() {
    NodeType t;
    t.id = "test_noise";
    t.display_name = "test_noise";
    t.is_format_sensitive = true;  // opt in to the post-process
    Socket out; out.name = "color"; out.type = SocketType::Vec4;
    t.outputs.push_back(out);
    t.glsl_function = "vec4 node_test_noise(vec2 uv) { return vec4(0.5, 0.5, 0.5, 1.0); }";
    return t;
}

NodeLibrary make_noise_library() {
    NodeLibrary lib;
    lib.add_public(make_noise_type());
    return lib;
}

} // namespace

TEST(EmitNodeShader, FormatSensitiveNodeEmitsTSFormatDefine) {
    auto lib = make_noise_library();
    Graph g;
    NodeInstance inst; inst.id = 1; inst.type_id = "test_noise";
    inst.format_override = ChannelFormat::Mono;
    g.nodes.push_back(inst);
    g.output_node = 1;
    auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
    auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);

    const std::string& glsl = c.pass_plan.passes[0].shader_glsl;
    EXPECT_NE(glsl.find("#define TS_FORMAT 0"), std::string::npos)
        << "Mono format must emit #define TS_FORMAT 0; got:\n" << glsl;
}

TEST(EmitNodeShader, DifferentFormatsEmitDifferentGLSL) {
    auto lib = make_noise_library();

    auto build = [&](ChannelFormat fmt) {
        Graph g;
        NodeInstance inst; inst.id = 1; inst.type_id = "test_noise";
        inst.format_override = fmt;
        g.nodes.push_back(inst);
        g.output_node = 1;
        auto r = validate_graph(g, lib); EXPECT_TRUE(r.success);
        auto c = GraphCompiler::compile(r.ir, lib); EXPECT_TRUE(c.success);
        return c.pass_plan.passes[0].shader_glsl;
    };

    const std::string mono  = build(ChannelFormat::Mono);
    const std::string uv    = build(ChannelFormat::UV);
    const std::string rgb   = build(ChannelFormat::RGB);
    const std::string rgba  = build(ChannelFormat::RGBA);
    const std::string meta  = build(ChannelFormat::Metadata);

    // All five define values must be present in their respective source.
    EXPECT_NE(mono.find("#define TS_FORMAT 0"), std::string::npos);
    EXPECT_NE(uv.find(  "#define TS_FORMAT 1"), std::string::npos);
    EXPECT_NE(rgb.find( "#define TS_FORMAT 2"), std::string::npos);
    EXPECT_NE(rgba.find("#define TS_FORMAT 3"), std::string::npos);
    EXPECT_NE(meta.find("#define TS_FORMAT 5"), std::string::npos);

    // All five sources must be distinct (this is the variant-differentiation
    // guarantee: same node, different format → different .spv input bytes).
    std::set<std::string> sources{mono, uv, rgb, rgba, meta};
    EXPECT_EQ(sources.size(), 5u);
}

TEST(EmitNodeShader, MonoBranchCollapsesAllChannelsToR) {
    auto lib = make_noise_library();
    Graph g;
    NodeInstance inst; inst.id = 1; inst.type_id = "test_noise";
    inst.format_override = ChannelFormat::Mono;
    g.nodes.push_back(inst);
    g.output_node = 1;
    auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
    auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);

    const std::string& glsl = c.pass_plan.passes[0].shader_glsl;
    // The Mono branch must replicate .r across all four channels. This is
    // the exact line that shaderc will compile into the .spv — if it
    // changes, the cache key → .spv mapping changes too.
    EXPECT_NE(glsl.find("vec4(result.r, result.r, result.r, 1.0)"),
              std::string::npos)
        << "Mono format must emit all-channels-to-.r collapse; got:\n"
        << glsl;
}

TEST(EmitNodeShader, UVBranchUsesGradientDropsNoise) {
    auto lib = make_noise_library();
    Graph g;
    NodeInstance inst; inst.id = 1; inst.type_id = "test_noise";
    inst.format_override = ChannelFormat::UV;
    g.nodes.push_back(inst);
    g.output_node = 1;
    auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
    auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);

    const std::string& glsl = c.pass_plan.passes[0].shader_glsl;
    // UV = .g (grad.x), .b (grad.y), 0, 1 — noise channel dropped.
    EXPECT_NE(glsl.find("vec4(result.g, result.b, 0.0, 1.0)"),
              std::string::npos)
        << "UV format must emit gradient-fold; got:\n" << glsl;
}

TEST(EmitNodeShader, NonFormatSensitiveNodeOmitsPostProcess) {
    // The 'passthrough' helper node from make_library() has is_format_sensitive
    // defaulted to false. Its emitted GLSL must NOT contain the post-process
    // (otherwise combiners would be collapsed to Mono incorrectly).
    auto lib = make_library();
    Graph g;
    NodeInstance inst; inst.id = 1; inst.type_id = "passthrough";
    inst.format_override = ChannelFormat::Mono;  // would be wrong if applied
    g.nodes.push_back(inst);
    g.output_node = 1;
    auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
    auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);

    const std::string& glsl = c.pass_plan.passes[0].shader_glsl;
    // TS_FORMAT define is still emitted (cheap, harmless) but the post-process
    // body must not appear.
    EXPECT_EQ(glsl.find("Format post-process"), std::string::npos)
        << "Non format-sensitive node must not emit the post-process; got:\n"
        << glsl;
    EXPECT_EQ(glsl.find("vec4(result.r, result.r, result.r, 1.0)"),
              std::string::npos)
        << "Mono collapse must not appear for combiners; got:\n" << glsl;
}

// ===========================================================================
// Edge cases: multi-output + format_sensitive, and specialization limits.
// ===========================================================================

TEST(EmitNodeShader, FormatSensitiveMultiOutputNodeOmitsPostProcess) {
    // Multi-output nodes (e.g. split_rgba) emit N imageStore calls in
    // the per-node path. The format post-process assumes a single `result`
    // variable; for multi-output we have `out0..outN-1`, so the post-process
    // block must be skipped. The graph compiler gates this on
    // `!multi_output`. If the gate breaks, all split_rgba-style nodes
    // would reference an undefined `result` and fail to compile.
    NodeLibrary lib;
    NodeType t;
    t.id = "test_split";
    t.display_name = "test_split";
    t.is_format_sensitive = true;     // opt in
    for (int i = 0; i < 4; ++i) {
        Socket s; s.name = "out" + std::to_string(i); s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    t.glsl_function =
        "void node_test_split(vec2 uv, out vec4 a, out vec4 b, out vec4 c, out vec4 d) {"
        " a = vec4(0); b = vec4(0); c = vec4(0); d = vec4(0); }";
    lib.add_public(t);

    Graph g;
    NodeInstance inst; inst.id = 1; inst.type_id = "test_split";
    inst.format_override = ChannelFormat::Mono;
    g.nodes.push_back(inst);
    g.output_node = 1;
    auto r = validate_graph(g, lib); ASSERT_TRUE(r.success);
    auto c = GraphCompiler::compile(r.ir, lib); ASSERT_TRUE(c.success);

    const std::string& glsl = c.pass_plan.passes[0].shader_glsl;
    EXPECT_EQ(glsl.find("Format post-process"), std::string::npos)
        << "multi-output node must skip the post-process block; got:\n" << glsl;
    EXPECT_EQ(glsl.find("vec4(result.r, result.r, result.r, 1.0)"),
              std::string::npos)
        << "Mono collapse must NOT appear for multi-output; got:\n" << glsl;
    // Confirm the per-output imageStore loop is present (proves we
    // actually exercised the multi_output branch, not the single path).
    EXPECT_NE(glsl.find("imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])]"),
              std::string::npos)
        << "multi-output loop should be in the GLSL; got:\n" << glsl;
}

TEST(ShaderVariantKey, SpecializationCountZeroProducesIdenticalHash) {
    // Two keys that differ only in unused specialization slots (count=0)
    // must hash and compare equal. The "garbage in slots > count" is
    // a real risk: if anyone forgets to gate the comparison on count,
    // a key with count=0 and zeroed slots would differ from one with
    // count=0 and arbitrary slots. (They shouldn't, but let's pin it.)
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 0;
    a.specialization.fill(0xDEADBEEF);  // garbage in unused slots
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 0;
    b.specialization.fill(0);            // zeroed unused slots

    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationCountIsPartOfTheHash) {
    // The count field is intentionally mixed into the hash (see
    // ShaderVariantKey::hash, line ~56). This means count=0 and
    // count=1 with the same slot 0 produce DIFFERENT hashes -- which
    // is correct, because a pipeline created with specialization[0]=42
    // is not the same variant as one with no specialization at all
    // (the GLSL `#define` would expand differently). If someone
    // "optimizes" by removing the count mix, this test fails.
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 0;
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 1;
    b.specialization[0] = 0;  // zero value shouldn't matter
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationSlotsBeyondIndexSevenAreIgnored) {
    // The array is std::array<uint32_t, 8>. The hash loop is
    // `i < specialization_count && i < 8`, so slots 8+ are never
    // read (and we can't write them either -- the array is fixed-size).
    // This means a count=8 key with slot 0=42 produces the same hash
    // as a count=8 key with slot 0=42 + anything in slots 1..7
    // zeroed -- because both mix all 8 slots. The test below pins
    // "all 8 slots are mixed" by showing that changing slot 7 alone
    // changes the hash.
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 8;
    a.specialization[7] = 1;
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 8;
    b.specialization[7] = 2;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationCountOneOnlyAffectsSlotZero) {
    // When count=1, only slot 0 is read. Two keys with the same slot 0
    // but different slot 1 must match. This pins the "compare up to
    // count" rule from operator== -- if someone changes it to
    // "compare all 8", the second key with a non-zero slot 1 would
    // fail to match, breaking per-instance state keys.
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 1;
    a.specialization[0] = 42;
    a.specialization[1] = 99;  // ignored
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 1;
    b.specialization[0] = 42;
    b.specialization[1] = 0;   // ignored

    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

