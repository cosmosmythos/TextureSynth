// =============================================================================
// node_worley — Tileable Cellular Noise
// =============================================================================
// Output:
//   .r = F1 (nearest distance, [0,1])
//   .g = F2 - F1 (edge mask, [0,1])
//   .b = cell ID hash, [0,1]  — feed to a gradient/color ramp downstream
//   .a = 1
// =============================================================================

vec4 node_worley(vec2 uv,
                 float period,
                 float octaves,
                 float lacunarity,
                 float gain,
                 float jitter,      // 0=grid, 1=full random
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

    TSWorley w = ts_fbm_worley(p, iper, ioct, lacunarity, gain,
                               clamp(jitter, 0.0, 1.0), iseed);

    return vec4(w.f1, clamp(w.f2 - w.f1, 0.0, 1.0), w.cell_id, 1.0);
}