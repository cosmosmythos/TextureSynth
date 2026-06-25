vec4 node_warp(vec2 uv, TSTexture image, TSTexture gradient,
                float intensity, float mode, float angle, float edge_wrap) {
    if (intensity < 1e-6) return Sample(image, uv);

    vec2 warpedUV = uv;

    if (mode < 0.5) {
        vec2 disp = Sample(gradient, uv).rg * 2.0 - 1.0;
        warpedUV = uv + disp * intensity;
    } else if (mode < 1.5) {
        vec2 dir = vec2(cos(angle), sin(angle));
        float h = Sample(gradient, uv).r;
        warpedUV = uv + dir * (h * 2.0 - 1.0) * intensity;
    } else if (mode < 2.5) {
        vec2 d = GetTexelSize(gradient);
        float hL = Sample(gradient, uv - vec2(d.x, 0.0)).r;
        float hR = Sample(gradient, uv + vec2(d.x, 0.0)).r;
        float hD = Sample(gradient, uv - vec2(0.0, d.y)).r;
        float hU = Sample(gradient, uv + vec2(0.0, d.y)).r;
        vec2 curl = vec2(hR - hL, hU - hD);
        warpedUV = uv + vec2(-curl.y, curl.x) * intensity;
    } else {
        vec2 d = GetTexelSize(gradient);
        float h = Sample(gradient, uv).r;
        float hL = Sample(gradient, uv - vec2(d.x, 0.0)).r;
        float hR = Sample(gradient, uv + vec2(d.x, 0.0)).r;
        float hD = Sample(gradient, uv - vec2(0.0, d.y)).r;
        float hU = Sample(gradient, uv + vec2(0.0, d.y)).r;
        vec2 grad = vec2(hR - hL, hU - hD);
        float len = length(grad);
        if (len > 1e-6) grad /= len;
        warpedUV = uv - grad * h * intensity;
    }

    if (edge_wrap < 0.5) {
        warpedUV = clamp(warpedUV, vec2(0.0), vec2(1.0));
    } else if (edge_wrap < 1.5) {
        warpedUV = fract(warpedUV);
    } else {
        warpedUV = abs(fract(warpedUV * 0.5) * 2.0 - 1.0);
    }

    return Sample(image, warpedUV);
}
