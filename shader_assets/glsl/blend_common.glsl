#ifndef TS_BLEND_COMMON
#define TS_BLEND_COMMON
// requires ts_rgb2hsv / ts_hsv2rgb from color_common.glsl

// ---- per-channel scalar primitives ------------------------
float ts_b_add       (float a, float b) { return min(a + b, 1.0); }
float ts_b_avg       (float a, float b) { return (a + b) * 0.5; }
float ts_b_darken    (float a, float b) { return min(a, b); }
float ts_b_lighten   (float a, float b) { return max(a, b); }
float ts_b_diff      (float a, float b) { return abs(a - b); }
float ts_b_mul       (float a, float b) { return a * b; }
float ts_b_screen    (float a, float b) { return 1.0 - (1.0 - a) * (1.0 - b); }
float ts_b_excl      (float a, float b) { return a + b - 2.0 * a * b; }
float ts_b_negation  (float a, float b) { return 1.0 - abs(1.0 - a - b); }
float ts_b_subtract  (float a, float b) { return max(a + b - 1.0, 0.0); }
float ts_b_linBurn   (float a, float b) { return max(a + b - 1.0, 0.0); }
float ts_b_linDodge  (float a, float b) { return min(a + b, 1.0); }
float ts_b_colorBurn (float a, float b) { return (b == 0.0) ? b : max(1.0 - (1.0 - a) / b, 0.0); }
float ts_b_colorDodge(float a, float b) { return (b == 1.0) ? b : min(a / (1.0 - b), 1.0); }
float ts_b_overlay   (float a, float b) { return (a < 0.5) ? 2.0*a*b : 1.0 - 2.0*(1.0-a)*(1.0-b); }
float ts_b_hardLight (float a, float b) { return ts_b_overlay(b, a); }
float ts_b_softLight (float a, float b) {
    return (b < 0.5) ? (2.0*a*b + a*a*(1.0 - 2.0*b))
                     : (sqrt(a) * (2.0*b - 1.0) + 2.0*a*(1.0 - b));
}
float ts_b_reflect   (float a, float b) { return (b == 1.0) ? b : min(a*a/(1.0-b), 1.0); }
float ts_b_glow      (float a, float b) { return ts_b_reflect(b, a); }
float ts_b_harmony   (float a, float b) { return min(a,b) - max(a,b) + 1.0; }
float ts_b_linLight  (float a, float b) { return (b < 0.5) ? ts_b_linBurn (a, 2.0*b) : ts_b_linDodge (a, 2.0*(b-0.5)); }
float ts_b_vividLight(float a, float b) { return (b < 0.5) ? ts_b_colorBurn(a, 2.0*b) : ts_b_colorDodge(a, 2.0*(b-0.5)); }
float ts_b_pinLight  (float a, float b) { return (b < 0.5) ? ts_b_darken  (a, 2.0*b) : ts_b_lighten  (a, 2.0*(b-0.5)); }
float ts_b_hardMix   (float a, float b) { return (ts_b_vividLight(a,b) < 0.5) ? 0.0 : 1.0; }

// ---- vec3 lift macro --------------------------------------------------------
#define TS_BLEND_VEC3(FN) vec3(FN(a.r,b.r), FN(a.g,b.g), FN(a.b,b.b))

// ---- HSL-style modes (whole-color, not per-channel) -------------------------
vec3 ts_b_hue       (vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(B.x, A.y, A.z)); }
vec3 ts_b_saturation(vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(A.x, B.y, A.z)); }
vec3 ts_b_color     (vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(B.x, B.y, A.z)); }
vec3 ts_b_luminosity(vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(A.x, A.y, B.z)); }

#endif