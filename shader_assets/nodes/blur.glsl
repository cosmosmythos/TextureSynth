vec4 node_blur(vec2 uv, TSTexture tex, float intensity, uint pass_index) {
    if (intensity < 1e-5) return Sample(tex, uv);

    float sigma = max(intensity * 50.0, 0.1);
    vec2 inv_res = 1.0 / vec2(GetSize(tex));
    vec2 dir = (pass_index == 0u) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    int radius = clamp(int(ceil(sigma * 3.0)), 1, 32);
    bool use_integrated = sigma < 0.7;

#define TSBLUR(o) texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[0])], samp_repeat), uv + (o))

    vec4 sum = vec4(0.0);
    float wsum = 0.0;

    for (int i = -radius; i <= radius; i++) {
        float fi = float(i);
        float w;
        if (use_integrated) {
            float p = (fi + 0.5) / (sigma * 1.41421356);
            float m = (fi - 0.5) / (sigma * 1.41421356);
            float tp = 1.0 / (1.0 + 0.3275911 * abs(p));
            float tm = 1.0 / (1.0 + 0.3275911 * abs(m));
            float ep = 1.0 - (((((1.06140543 * tp - 1.45315203) * tp + 1.42141374) * tp - 0.28449674) * tp + 0.25482959) * tp) * exp(-p * p);
            float em = 1.0 - (((((1.06140543 * tm - 1.45315203) * tm + 1.42141374) * tm - 0.28449674) * tm + 0.25482959) * tm) * exp(-m * m);
            w = (p >= 0.0 ? ep : -ep) - (m >= 0.0 ? em : -em);
        } else {
            w = exp(-fi * fi / (2.0 * sigma * sigma));
        }
        sum += TSBLUR(dir * fi * inv_res) * w;
        wsum += w;
    }

#undef TSBLUR

    return sum / wsum;
}
