vec4 node_remap(vec2 uv, vec4 color,
                float in_min, float in_max,
                float out_min, float out_max)
{
    return ts_remap(color, in_min, in_max, out_min, out_max);
}
