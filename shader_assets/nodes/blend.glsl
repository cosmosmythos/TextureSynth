// modes must match Enum order in blend.py

vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b, float mode) {
    int  m = int(mode + 0.5);
    float f = clamp(mask, 0.0, 1.0);
    vec3 r;
    if      (m ==  1) r = TS_BLEND_VEC3(ts_b_add);
    else if (m ==  2) r = TS_BLEND_VEC3(ts_b_mul);
    else if (m ==  3) r = TS_BLEND_VEC3(ts_b_screen);
    else if (m ==  4) r = TS_BLEND_VEC3(ts_b_overlay);
    else if (m ==  5) r = TS_BLEND_VEC3(ts_b_diff);
    else if (m ==  6) r = TS_BLEND_VEC3(ts_b_darken);
    else if (m ==  7) r = TS_BLEND_VEC3(ts_b_lighten);
    else if (m ==  8) r = TS_BLEND_VEC3(ts_b_colorBurn);
    else if (m ==  9) r = TS_BLEND_VEC3(ts_b_colorDodge);
    else if (m == 10) r = TS_BLEND_VEC3(ts_b_linBurn);
    else if (m == 11) r = TS_BLEND_VEC3(ts_b_linDodge);
    else if (m == 12) r = TS_BLEND_VEC3(ts_b_linLight);
    else if (m == 13) r = TS_BLEND_VEC3(ts_b_vividLight);
    else if (m == 14) r = TS_BLEND_VEC3(ts_b_pinLight);
    else if (m == 15) r = TS_BLEND_VEC3(ts_b_hardLight);
    else if (m == 16) r = TS_BLEND_VEC3(ts_b_softLight);
    else if (m == 17) r = TS_BLEND_VEC3(ts_b_hardMix);
    else if (m == 18) r = TS_BLEND_VEC3(ts_b_excl);
    else if (m == 19) r = TS_BLEND_VEC3(ts_b_subtract);
    else if (m == 20) r = TS_BLEND_VEC3(ts_b_avg);
    else if (m == 21) r = TS_BLEND_VEC3(ts_b_negation);
    else if (m == 22) r = TS_BLEND_VEC3(ts_b_reflect);
    else if (m == 23) r = TS_BLEND_VEC3(ts_b_glow);
    else if (m == 24) r = TS_BLEND_VEC3(ts_b_harmony);
    else if (m == 25) r = ts_b_hue       (a.rgb, b.rgb);
    else if (m == 26) r = ts_b_saturation(a.rgb, b.rgb);
    else if (m == 27) r = ts_b_color     (a.rgb, b.rgb);
    else if (m == 28) r = ts_b_luminosity(a.rgb, b.rgb);
    else              r = mix(a.rgb, b.rgb, f);

    if (m != 0) r = mix(a.rgb, r, f);
    return vec4(clamp(r, 0.0, 1.0), mix(a.a, b.a, f));
}