vec4 node_worley(vec2 uv,
                 float period,
                 float octaves,
                 float lacunarity,
                 float roughness,
                 float jitter,
                 float speed,
                 float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    float jit = clamp(jitter, 0.0, 1.0);

    vec2 p = fract(uv) * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed).f1;
    float g = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed + 379u).f1;
    float b = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed + 757u).f1;

    return vec4(r, g, b, 1.0);
}
