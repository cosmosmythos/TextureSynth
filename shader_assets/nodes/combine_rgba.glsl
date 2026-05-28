// Combine RGBA - combine four grayscale inputs into a single RGBA color.
// Each input's RED channel is sampled as the scalar value for that slot.

vec4 node_combine_rgba(vec2 uv, vec4 r, vec4 g, vec4 b, vec4 a) {
    return vec4(r.r, g.r, b.r, a.r);
}