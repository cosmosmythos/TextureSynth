vec4 node_white(vec2 uv,
                float resolution,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int ires = max(int(round(resolution)), 1);
    vec2 p = uv * float(ires);
    vec3 n = ts_white_pcg_vec3(p, ires, iseed);
    return vec4(n, 1.0);
}
