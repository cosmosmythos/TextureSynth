---
name: glsl
description: Use when writing, modifying, or debugging GLSL compute shaders for TextureSynth noise nodes — covers portability bugs, integer modulo, hash functions, wrapping, and GLSL spec gotchas.
---

## What I do
- Portability-aware GLSL writing for noise/compute shaders
- Diagnose and fix GPU vs CPU divergence (hash mismatches, wrapping bugs)
- Audit `*.glsl` files in `shader_assets/` for spec compliance
- Document GLSL driver quirks and workarounds

## When to use me
- Writing new noise node GLSL under `shader_assets/nodes/` or `shader_assets/glsl/`
- Modifying shared noise primitives in `shader_assets/`
- Debugging GPU vs CPU output divergence
- Auditing shaders for portability before deployment
- Fixing tiling, wraparound, or hash-seed issues

## Hashing guidance

### Use `uint` arithmetic for hashes
Prefer `uint`-based hashes throughout the lattice pipeline.
- Avoid implicit signed conversions
- Avoid mixing `int` and `uint` unnecessarily
- Keep the seed in `uint`
- Convert to float only at the final normalization step


## Known Portability Bugs

### GLSL Integer Modulo (`%`) — CRITICAL
**Bug**: Integer `%` is not portable for negative operands in GLSL. Some drivers have produced incorrect results, so negative wrapping must not rely on `%`.
**Fix**: Use Euclidean wrapping via floating-point `mod()`:

```glsl
int ts_wrap(int v, int per) {
    return int(mod(float(v), float(per)));
}
```

**Where this matters**: Any noise function that wraps cell coordinates for tiling (value, perlin, simplex, worley, gabor). The wrapping only triggers for negative `pi` values (cells at the boundary where `floor(p) < 0`).

**How to detect**: If tiling fails at non-power-of-2 periods but passes at power-of-2, suspect this bug. Power-of-2 periods mask it because the wrap result happens to be the same.

**Affected files**:
- `shader_assets/glsl/noise_common.glsl:161` — `ts_wrap()` (FIXED)
- `fused_perlin_invert.glsl:237` — `ts_wrap()` (FIXED)

### Other Gotchas

| Issue | Pattern | Fix |
|---|---|---|
| `int(float)` truncation | `int p = int(3.7);` → `3` | Use `int(round(v))` for intentional rounding |
| `float` precision at large ints | `float(0xFFFFFFFF)` → wrong | Use `floatBitsToFloat()` or keep in uint |
| `mod()` vs `%` semantics | `mod(-1, 7)` → `6`, `(-1) % 7` → `-1` (spec) or `-4` (buggy) | Always use `mod()` for wrapping |
| `floor()` on exact integers | `floor(3.0)` → `3.0` (correct) but `floor(2.999999)` → `2.0` | Ensure period multiplication before floor, not after |


## Bindless sampler style

GLSL bindless sampler constructors (`sampler2D(u_sampled[...], samp)`) MUST appear at the point of use in `texture()` calls — they CANNOT be assigned to local variables or passed as function arguments.

**Pattern:** use a `#define` macro to avoid repetition:

```glsl
#define TSBLUR(o) texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[0])], samp_repeat), uv + (o))

    return TSBLUR(vec2(0))  * 0.2270270270
         + TSBLUR(+off1)    * 0.3162162162
         + TSBLUR(-off1)    * 0.3162162162;

#undef TSBLUR
```

**Rules:**
- Macro name: `TS` prefix + descriptive suffix (e.g. `TSBLUR`, `TSNOISE`)
- Always `#undef` after use
- Little to no comments — let clear naming carry it
- Alignment: align `=` and `+` operators for readability

## Verification checklist
- [ ] No integer `%` on negative values — use `int(mod(float(v), float(per)))` for wrapping
- [ ] `ts_wrap()` used for all cell coordinate wrapping in tiling noise
- [ ] Time-shift tiling test passes for non-power-of-2 periods (3, 5, 7, 11)
- [ ] No `int(float)` truncation where `round()` is intended
- [ ] Hash functions use `uint` arithmetic throughout (no implicit sign extension)
- [ ] Bindless samplers use `#define` macros, not locals or function args
