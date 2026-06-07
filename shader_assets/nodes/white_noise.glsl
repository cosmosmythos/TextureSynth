// White Noise using PCG2D hash — Multi-Seed RGB.
// PCG operates on full uint32, giving 2^32 distinct values — no 289-value
// banding at any resolution. Output already in [0,1]; ts_to_unit not needed.
vec4 node_white(vec2 uv,
                float resolution,
                float offsetX,
                float offsetY,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int ires = max(int(round(resolution)), 1);
    vec2 p = uv * float(ires) + vec2(offsetX, offsetY);
    float r = ts_white_pcg(p, ires, iseed);
    float g = ts_white_pcg(p, ires, iseed + 379u);
    float b = ts_white_pcg(p, ires, iseed + 757u);
    return vec4(r, g, b, 1.0);
}