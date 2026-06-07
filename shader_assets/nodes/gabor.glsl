// =============================================================================
// node_gabor — Anisotropic Frequency-Controlled Noise (Multi-Seed RGB)
// =============================================================================
// Each channel is an independent gabor noise with a different seed offset.
// =============================================================================

vec4 node_gabor(vec2 uv,
                float period,
                float octaves,
                float lacunarity,
                float gain,
                float frequency,
                float bandwidth,
                float anisotropy,
                float angle,
                float offsetX,
                float offsetY,
                float speed,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    int ioct = clamp(int(round(octaves)), 1, 8);
    float aniso = clamp(anisotropy, 0.0, 1.0);

    vec2 p = uv * float(iper)
           + vec2(offsetX, offsetY)
           + vec2(pc.time * speed);

    float r = ts_fbm_gabor(p, iper, ioct, lacunarity, gain,
                           frequency, bandwidth, aniso, angle, iseed);
    float g = ts_fbm_gabor(p, iper, ioct, lacunarity, gain,
                           frequency, bandwidth, aniso, angle, iseed + 379u);
    float b = ts_fbm_gabor(p, iper, ioct, lacunarity, gain,
                           frequency, bandwidth, aniso, angle, iseed + 757u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), 1.0);
}