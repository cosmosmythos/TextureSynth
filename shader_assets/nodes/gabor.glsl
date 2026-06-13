vec4 node_gabor(vec2 uv,
                float period,
                float octaves,
                float lacunarity,
                float roughness,
                float frequency,
                float bandwidth,
                float anisotropy,
                float angle,
                float speed,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_pow2(period);
    float aniso = clamp(anisotropy, 0.0, 1.0);

    vec2 p = uv * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed);
    float g = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed + 379u);
    float b = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed + 757u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), 1.0);
}
