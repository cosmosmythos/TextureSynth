// =============================================================================
// node_gabor — Anisotropic Frequency-Controlled Noise
// =============================================================================
// Use for wood grain, brushed metal, fabric, directional fibers.
// =============================================================================

vec4 node_gabor(vec2 uv,
                float period,
                float octaves,
                float lacunarity,
                float gain,
                float frequency,    // cycles per cell, ~3..20
                float bandwidth,    // Gaussian falloff, ~2..8
                float anisotropy,   // 0=isotropic, 1=fully aligned
                float angle,        // radians
                float offsetX,
                float offsetY,
                float speed,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    int ioct = clamp(int(round(octaves)), 1, 8);

    vec2 p = uv * float(iper)
           + vec2(offsetX, offsetY)
           + vec2(pc.time * speed);

    float n = ts_fbm_gabor(p, iper, ioct, lacunarity, gain,
                           frequency, bandwidth,
                           clamp(anisotropy, 0.0, 1.0), angle, iseed);

    float c = ts_to_unit(n);
    return vec4(c, c, c, 1.0);
}