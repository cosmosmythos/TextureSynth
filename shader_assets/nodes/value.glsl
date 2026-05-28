vec4 node_value(vec2 uv,
                float period, float octaves, float lacunarity, float gain,
                float offsetX, float offsetY, float speed, float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    int ioct = clamp(int(round(octaves)), 1, 8);

    vec2 p = uv * float(iper)
           + vec2(offsetX, offsetY)
           + vec2(pc.time * speed);

    vec2 grad;
    float n = ts_fbm_value(p, iper, ioct, lacunarity, gain, iseed, grad);

    return vec4(ts_to_unit(n), ts_to_unit(grad.x), ts_to_unit(grad.y), 1.0);
}