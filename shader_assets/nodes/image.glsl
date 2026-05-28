vec4 node_image(vec2 uv, TSTexture tex,
                float u_offset, float v_offset,
                float u_scale,  float v_scale)
{
    vec2 t = (uv - vec2(u_offset, v_offset)) / vec2(u_scale, v_scale);
    return Sample(tex, t);
}
