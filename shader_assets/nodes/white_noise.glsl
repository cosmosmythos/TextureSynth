vec4 node_white(vec2 uv,
                float resolution,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int ires = max(int(round(resolution)), 1);
    vec2 p = uv * float(ires);

    float r = ts_white_pcg(p, ires, iseed);
    float g = ts_white_pcg(p, ires, iseed + 379u);
    float b = ts_white_pcg(p, ires, iseed + 757u);
    float a = ts_white_pcg(p, ires, iseed + 1013u);

    return vec4(r, g, b, a);
}
