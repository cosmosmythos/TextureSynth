vec4 node_invert(vec2 uv, vec4 color, float amount) {
    vec3 inv = vec3(1.0) - color.rgb;
    return vec4(mix(color.rgb, inv, ts_saturate(amount)), color.a);
}
