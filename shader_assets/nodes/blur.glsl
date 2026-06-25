// blur — Separable Gaussian blur (2-pass: horizontal then vertical).
// Uses bilinear-optimized 9-tap kernel from glsl-fast-gaussian-blur (MIT).
// ts_pass_index is a specialization constant: 0=horizontal, 1=vertical.
// The GLSL compiler eliminates the dead branch at compile time.
// NOTE: sampler2D(u_sampled[...], samp) must be inlined at each texture()
// call — bindless samplers can't be assigned to locals or passed as args.

vec4 node_blur(vec2 uv, TSTexture tex, float intensity) {
    if (intensity < 1e-5) return Sample(tex, uv);

    vec2 res = vec2(GetSize(tex));
    float sigma = intensity * 5.0;

    // ts_pass_index is specialization constant: 0=H, 1=V.
    // Compiler eliminates the dead branch.
    vec2 direction = (ts_pass_index == 0u)
        ? vec2(sigma, 0.0)   // horizontal
        : vec2(0.0, sigma);  // vertical

    // Bilinear-optimized 9-tap Gaussian (from glsl-fast-gaussian-blur, MIT).
    // 5 texture reads instead of 9 — bilinear hardware gives in-between pixels.
    vec2 off1 = vec2(1.3846153846) * direction;
    vec2 off2 = vec2(3.2307692308) * direction;
    vec2 inv_res = 1.0 / res;

#define BLUR_IMG sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[0])], samp_repeat)

    vec4 color = vec4(0.0);
    color += texture(BLUR_IMG, uv)                    * 0.2270270270;
    color += texture(BLUR_IMG, uv + off1 * inv_res)   * 0.3162162162;
    color += texture(BLUR_IMG, uv - off1 * inv_res)   * 0.3162162162;
    color += texture(BLUR_IMG, uv + off2 * inv_res)   * 0.0702702703;
    color += texture(BLUR_IMG, uv - off2 * inv_res)   * 0.0702702703;

#undef BLUR_IMG

    return color;
}
