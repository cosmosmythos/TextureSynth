float ts_levels_single(float v, float in_low, float in_mid, float in_high,
                       float out_low, float out_high) {
    float range = in_high - in_low;
    float x;
    if (range >= 0.0) {
        x = clamp((v - in_low) / max(range, 1e-6), 0.0, 1.0);
    } else {
        x = clamp((in_low - v) / max(-range, 1e-6), 0.0, 1.0);
    }

    float t = clamp(in_mid, 1e-4, 1.0 - 1e-4);
    float p = log(0.5) / log(t);

    return pow(x, p) * (out_high - out_low) + out_low;
}

vec4 node_levels(vec2 uv, vec4 color,
                 float in_low_l, float in_mid_l, float in_high_l,
                 float out_low_l, float out_high_l,
                 float in_low_r, float in_mid_r, float in_high_r,
                 float out_low_r, float out_high_r,
                 float in_low_g, float in_mid_g, float in_high_g,
                 float out_low_g, float out_high_g,
                 float in_low_b, float in_mid_b, float in_high_b,
                 float out_low_b, float out_high_b,
                 float in_low_a, float in_mid_a, float in_high_a,
                 float out_low_a, float out_high_a,
                 float channel_mode)
{
    int m = int(channel_mode + 0.5);

    float r, g, b, a;

    if (m == 0) {
        // Luminance: L settings override R, G, B
        r = ts_levels_single(color.r, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
        g = ts_levels_single(color.g, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
        b = ts_levels_single(color.b, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
    } else {
        // Per-channel: each uses its own settings
        r = ts_levels_single(color.r, in_low_r, in_mid_r, in_high_r, out_low_r, out_high_r);
        g = ts_levels_single(color.g, in_low_g, in_mid_g, in_high_g, out_low_g, out_high_g);
        b = ts_levels_single(color.b, in_low_b, in_mid_b, in_high_b, out_low_b, out_high_b);
    }

    a = ts_levels_single(color.a, in_low_a, in_mid_a, in_high_a, out_low_a, out_high_a);

    return vec4(r, g, b, a);
}
