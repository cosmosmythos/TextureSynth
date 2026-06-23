#ifndef TS_BLEND_COMMON
#define TS_BLEND_COMMON

// a = foreground, b = background. Each fn returns the raw mode result.
#define TS_BLEND_VEC3(FN) vec3(FN(a.r,b.r), FN(a.g,b.g), FN(a.b,b.b))

float ts_blend_add (float a, float b) { return a + b; }
float ts_blend_sub (float a, float b) { return a - b; }
float ts_blend_mul (float a, float b) { return (a < 0.0 && b < 0.0) ? a : a * b; }
float ts_blend_min (float a, float b) { return min(a, b); }
float ts_blend_max (float a, float b) { return max(a, b); }
float ts_blend_avg (float a, float b) { return (a + b) * 0.5; }

float ts_blend_color_burn(float a, float b) { return 1.0 - (1.0 - b) / max(a, 1.0 - b); }

float ts_blend_overlay(float a, float b) {
    if (a < 0.0 || b < 0.0) return a;
    return (2.0 * b < 1.0) ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
}

float ts_blend_screen(float a, float b) {
    return (a > 0.0 && a < 1.0 && b > 0.0 && b < 1.0) ? a + b - a * b : max(a, b);
}

float ts_blend_color_dodge(float a, float b) { return b / max(1.0 - a, b); }

float ts_blend_soft_light(float a, float b) {
    return (a * b < 1.0) ? 2.0 * a * b + b * (1.0 - a * b) : 2.0 * a * b;
}

float ts_blend_hard_light(float a, float b) {
    if (2.0 * a < 1.0) return 2.0 * a * b;
    if (a < 1.0 && b < 1.0) return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
    return 0.0;
}

float ts_blend_divide(float a, float b) { return (b > 0.0) ? a / b : a; }
float ts_blend_diff (float a, float b) { return abs(a - b); }

float ts_blend_exclusion(float a, float b) {
    if (a < 0.0 || b < 0.0) return a;
    return a + b - 2.0 * a * b;
}

#endif
