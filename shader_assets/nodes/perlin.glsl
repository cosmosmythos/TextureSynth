// =============================================================================
// node_perlin — Tileable Classic Perlin FBM
// =============================================================================
// Output: vec4(noise, ∂n/∂x, ∂n/∂y, 1)  — all channels remapped to [0,1].
// Connect .r for height/luminance, .gb for downstream domain-warp or normal map.
// =============================================================================

vec4 node_perlin(vec2 uv,
                 float period,       // positive integer; lattice tile size
                 float octaves,      // 1..8
                 float lacunarity,   // 2.0 standard (integer for exact tiling)
                 float gain,         // 0..1
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

    vec2 grad;
    float n = ts_fbm_perlin(p, iper, ioct, lacunarity, gain, iseed, grad);

    return vec4(ts_to_unit(n), ts_to_unit(grad.x), ts_to_unit(grad.y), 1.0);
}