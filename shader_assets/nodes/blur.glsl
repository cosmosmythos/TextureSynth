vec4 node_blur(vec2 uv, TSTexture tex, float intensity) {
    if (intensity < 1e-5) return Sample(tex, uv);

    vec2 ts = vec2(GetSize(tex));
    float radius = clamp(intensity * 10.0, 1.0, 20.0);
    int M = int(ceil(radius));

    vec2 sigma = vec2(radius * 0.3);
    vec4 sum = vec4(0.0);
    float wsum = 0.0;

    for (int j = -M; j <= M; j++) {
        for (int i = -M; i <= M; i++) {
            float fi = float(i);
            float fj = float(j);
            float w = exp(-(fi*fi/(2.0*sigma.x*sigma.x)
                         + fj*fj/(2.0*sigma.y*sigma.y)));
            vec2 off = vec2(fi, fj) / ts;
            sum += Sample(tex, uv + off) * w;
            wsum += w;
        }
    }

    return sum / wsum;
}
