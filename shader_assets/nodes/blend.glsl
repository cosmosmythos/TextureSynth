// a = foreground, b = background. mask: 0 -> B, 1 -> mode result.

vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b, float mode) {
    int m = int(mode + 0.5);
    float f = clamp(mask, 0.0, 1.0);
    vec3 r = a.rgb;
    switch (m) {
        case 1:  r = TS_BLEND_VEC3(ts_blend_add);          break;
        case 2:  r = TS_BLEND_VEC3(ts_blend_sub);          break;
        case 3:  r = TS_BLEND_VEC3(ts_blend_mul);          break;
        case 4:  r = TS_BLEND_VEC3(ts_blend_min);          break;
        case 5:  r = TS_BLEND_VEC3(ts_blend_max);          break;
        case 6:  r = TS_BLEND_VEC3(ts_blend_avg);          break;
        case 7:  r = TS_BLEND_VEC3(ts_blend_color_burn);   break;
        case 8:  r = TS_BLEND_VEC3(ts_blend_overlay);      break;
        case 9:  r = TS_BLEND_VEC3(ts_blend_screen);       break;
        case 10: r = TS_BLEND_VEC3(ts_blend_color_dodge);  break;
        case 11: r = TS_BLEND_VEC3(ts_blend_soft_light);   break;
        case 12: r = TS_BLEND_VEC3(ts_blend_hard_light);   break;
        case 13: r = TS_BLEND_VEC3(ts_blend_divide);       break;
        case 14: r = TS_BLEND_VEC3(ts_blend_diff);         break;
        case 15: r = TS_BLEND_VEC3(ts_blend_exclusion);    break;
    }
    return vec4(mix(b.rgb, r, f), mix(b.a, a.a, f));
}
