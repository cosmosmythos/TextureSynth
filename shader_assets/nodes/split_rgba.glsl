// Split RGBA - split a color texture into four grayscale channels.

void node_split_rgba(vec2 uv, vec4 color,
                     out vec4 r_out,
                     out vec4 g_out,
                     out vec4 b_out,
                     out vec4 a_out)
{
    r_out = vec4(color.r, color.r, color.r, 1.0);
    g_out = vec4(color.g, color.g, color.g, 1.0);
    b_out = vec4(color.b, color.b, color.b, 1.0);
    a_out = vec4(color.a, color.a, color.a, 1.0);
}
