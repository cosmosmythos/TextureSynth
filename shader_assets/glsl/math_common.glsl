#ifndef TS_MATH_COMMON
#define TS_MATH_COMMON

float ts_saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2  ts_saturate(vec2  x) { return clamp(x, 0.0, 1.0); }
vec3  ts_saturate(vec3  x) { return clamp(x, 0.0, 1.0); }
vec4  ts_saturate(vec4  x) { return clamp(x, 0.0, 1.0); }

float ts_remap(float v, float a, float b, float c, float d) {
    float denom = b - a;
    if (abs(denom) < 1e-8) return c;
    return c + (d - c) * (v - a) / denom;
}

#endif