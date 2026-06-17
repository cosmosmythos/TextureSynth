vec4 node_remap(vec2 uv, vec4 color,
                float in_min, float in_max,
                float out_min, float out_max,
                float channel_mode)
{
    int m = int(channel_mode + 0.5);
    if (m == 0) {
        return vec4(ts_remap(color.r, in_min, in_max, out_min, out_max),
                    color.g, color.b, color.a);
    } else if (m == 1) {
        return vec4(ts_remap(color.r, in_min, in_max, out_min, out_max),
                    ts_remap(color.g, in_min, in_max, out_min, out_max),
                    color.b, color.a);
    } else if (m == 2) {
        vec3 rg = ts_remap(color.rgb, in_min, in_max, out_min, out_max);
        return vec4(rg, color.a);
    }
    return ts_remap(color, in_min, in_max, out_min, out_max);
}
