// =============================================================================
// node_simplex — Tileable Simplex FBM (Multi-Seed RGB)
// =============================================================================
// Output: vec4(noise_R, noise_G, noise_B, 1)  — Houdini convention.
// Each channel is an independent simplex noise with a different seed offset.
// =============================================================================

vec4 node_simplex(vec2 uv,
                  float period,
                  float octaves,
                  float lacunarity,
                  float gain,
                  float offsetX,
                  float offsetY,
                  float speed,
                  float rotation,
                  float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iperX = ts_period_int(period);
    int iperY = ts_period_even(period);
    int ioct  = clamp(int(round(octaves)), 1, 8);

    vec2 p = uv * float(iperX)
           + vec2(offsetX, offsetY)
           + vec2(pc.time * speed);

    ivec2 per = ivec2(iperX, iperY);

    float r = ts_fbm_simplex(p, per, ioct, lacunarity, gain, rotation, iseed);
    float g = ts_fbm_simplex(p, per, ioct, lacunarity, gain, rotation, iseed + 379u);
    float b = ts_fbm_simplex(p, per, ioct, lacunarity, gain, rotation, iseed + 757u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), 1.0);
}