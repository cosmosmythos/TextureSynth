// =============================================================================
// node_worley — Tileable Cellular Noise (Multi-Seed RGB)
// =============================================================================
// Output: vec4(worley_R, worley_G, worley_B, 1)  — Houdini convention.
// Each channel is an independent worley F1 field with a different seed offset.
// =============================================================================

vec4 node_worley(vec2 uv,
                 float period,
                 float octaves,
                 float lacunarity,
                 float gain,
                 float jitter,
                 float offsetX,
                 float offsetY,
                 float speed,
                 float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    int ioct = clamp(int(round(octaves)), 1, 8);
    float jit = clamp(jitter, 0.0, 1.0);

    vec2 p = uv * float(iper)
           + vec2(offsetX, offsetY)
           + vec2(pc.time * speed);

    float r = ts_fbm_worley(p, iper, ioct, lacunarity, gain, jit, iseed).f1;
    float g = ts_fbm_worley(p, iper, ioct, lacunarity, gain, jit, iseed + 379u).f1;
    float b = ts_fbm_worley(p, iper, ioct, lacunarity, gain, jit, iseed + 757u).f1;

    return vec4(r, g, b, 1.0);
}