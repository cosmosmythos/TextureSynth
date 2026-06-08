vec4 node_invert(vec2 uv, vec4 color, float mask) {
    vec3 inv = vec3(1.0) - color.rgb;
    vec4 result = vec4(inv, color.a);
    return mix(color, result, clamp(mask, 0.0, 1.0));
}
