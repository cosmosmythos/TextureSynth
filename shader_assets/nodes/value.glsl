vec4 node_value(vec2 uv,
                float period, float octaves, float lacunarity, float roughness,
                float speed, float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);

    vec2 p = uv * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_value(p, iper, octaves, lacunarity, roughness, iseed);
    float g = ts_fbm_value(p, iper, octaves, lacunarity, roughness, iseed + 379u);
    float b = ts_fbm_value(p, iper, octaves, lacunarity, roughness, iseed + 757u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), 1.0);
}
