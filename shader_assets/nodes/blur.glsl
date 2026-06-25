vec4 node_blur(vec2 uv, TSTexture tex, float intensity) {
    if (intensity < 1e-5) return Sample(tex, uv);

    vec2 inv_res = 1.0 / vec2(GetSize(tex));
    float sigma  = intensity * 50.0;
    vec2 dir     = (ts_pass_index == 0u) ? vec2(sigma, 0.0) : vec2(0.0, sigma);

    vec2 off1 = 1.3846153846 * dir * inv_res;
    vec2 off2 = 3.2307692308 * dir * inv_res;

#define TSBLUR(o) texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[0])], samp_repeat), uv + (o))

    return TSBLUR(vec2(0))  * 0.2270270270
         + TSBLUR(+off1)    * 0.3162162162
         + TSBLUR(-off1)    * 0.3162162162
         + TSBLUR(+off2)    * 0.0702702703
         + TSBLUR(-off2)    * 0.0702702703;

#undef TSBLUR
}
