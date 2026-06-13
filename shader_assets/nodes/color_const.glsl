vec4 node_color_const(vec2 uv, float mode, float r, float g, float b, float a) {
    if (mode < 0.5) return vec4(r, r, r, 1.0);
    return vec4(r, g, b, a);
}