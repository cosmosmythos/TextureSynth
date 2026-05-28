// =============================================================================
// node_simplex — Tileable Simplex FBM with Analytical Derivatives (psrdnoise2)
// =============================================================================
// Output: vec4(noise, ∂n/∂x, ∂n/∂y, 1)  — all remapped to [0,1].
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

    vec2 grad;
    float n = ts_fbm_simplex(p, ivec2(iperX, iperY), ioct, lacunarity, gain, rotation, iseed, grad);

    return vec4(ts_to_unit(n), ts_to_unit(grad.x), ts_to_unit(grad.y), 1.0);
}