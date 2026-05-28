// White Noise using PCG2D hash.
// PCG operates on full uint32, giving 2^32 distinct values — no 289-value
// banding at any resolution.  Output already in [0,1]; ts_to_unit not needed.
vec4 node_white(vec2 uv,
                float resolution,   // pixel grid size; integer
                float offsetX,
                float offsetY,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int ires = max(int(round(resolution)), 1);
    vec2 p = uv * float(ires) + vec2(offsetX, offsetY);
    float n = ts_white_pcg(p, ires, iseed);
    return vec4(n, n, n, 1.0);
}