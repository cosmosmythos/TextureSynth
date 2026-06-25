vec4 node_warp(vec2 uv, TSTexture image, TSTexture gradient, float intensity) {
    if (intensity < 1e-6) return Sample(image, uv);

    vec2 disp = Sample(gradient, uv).rg * 2.0 - 1.0;
    vec2 warpedUV = uv + disp * intensity;
    return Sample(image, warpedUV);
}
