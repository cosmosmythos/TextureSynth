vec4 node_value(vec2 uv,
                float period, float octaves, float lacunarity, float roughness,
                float speed, float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);

    vec2 p = uv * float(iper)
           + vec2(pc.time * speed);

    vec3 n = ts_fbm_value_vec3(p, iper, octaves, lacunarity, roughness, iseed);

    return vec4(ts_to_unit(n.x), ts_to_unit(n.y), ts_to_unit(n.z), 1.0);
}
