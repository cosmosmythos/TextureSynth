#ifndef TS_REMAP
#define TS_REMAP

float ts_remap(float v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return out_min;
    return out_min + (out_max - out_min) * (v - in_min) / denom;
}

vec2 ts_remap(vec2 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec2(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec3 ts_remap(vec3 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec3(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec4 ts_remap(vec4 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec4(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

#endif
