// Grayscale conversion.
//
// mode 0 - Luminance (Rec.709): 0.2126*R + 0.7152*G + 0.0722*B
//           Perceptually correct; matches human eye sensitivity.
//           Same formula used by lygia rgb2luma and Photoshop Desaturate.
// mode 1 - Average: (R+G+B)/3
//           Simple, uniform weight. Predictable for procedural patterns.
// mode 2 - Max channel (Value from HSV)
//           Preserves the brightest channel; useful for lightness maps.
// mode 3 - Min channel
//           Preserves the darkest channel; useful for shadow masks.
vec4 node_grayscale(vec2 uv, vec4 color, float mode, float mask) {
    int m = int(mode + 0.5);
    float v;
    if      (m == 1) v = (color.r + color.g + color.b) * 0.33333333;
    else if (m == 2) v = max(color.r, max(color.g, color.b));
    else if (m == 3) v = min(color.r, min(color.g, color.b));
    else             v = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722)); // Rec.709
    vec4 result = vec4(v, v, v, color.a);
    return mix(color, result, clamp(mask, 0.0, 1.0));
}