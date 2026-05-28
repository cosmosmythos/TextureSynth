// Constant Color / Constant Value node.
// mode 0 - Grayscale: outputs (r, r, r, 1.0), r holds the [0,1] scalar.
// mode 1 - RGBA:      outputs (r, g, b, a) directly.
// Python UI overrides draw_buttons with a color picker (Sverchok pattern),
// so artists never see raw r/g/b/a sliders; get_parameters() extracts them.
vec4 node_color_const(vec2 uv, float mode, float r, float g, float b, float a) {
    if (mode < 0.5) return vec4(r, r, r, 1.0);
    return vec4(r, g, b, a);
}