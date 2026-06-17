vec4 node_shuffle(vec2 uv, vec4 color,
                  float r_src, float g_src,
                  float b_src, float a_src)
{
    float sources[4] = float[4](color.r, color.g, color.b, color.a);
    return vec4(
        sources[int(r_src + 0.5)],
        sources[int(g_src + 0.5)],
        sources[int(b_src + 0.5)],
        sources[int(a_src + 0.5)]
    );
}
