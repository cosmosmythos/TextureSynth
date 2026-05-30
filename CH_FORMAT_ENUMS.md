# ChannelFormat Enums — Future Plan

> *Vision document for how format_override should evolve from a global
> dropdown to a per-node, shader-aware configuration system.*

---

## Current State (the problem)

Every auto-generated node gets a `format_override` dropdown with 6 options:

```
[Auto | Mono | UV | RGB | RGBA | Integer]
```

This is defined on the **base class** `TextureSynthNode` (`ADDON/nodes/base.py:52`),
so **every node** shows it. This is wrong for several reasons:

**1. Conceptually meaningless on many nodes**
- What does "Mono" mean on a `color_const` node? It outputs a solid color.
  The format shouldn't be user-selectable — it's conceptually always RGBA.
- What does "UV" mean on a `split_rgba` R output? It's always Mono.
  The format is **fixed** by the node's purpose.

**2. The dropdown is a distraction**
- 80% of nodes should have NO format dropdown. It adds visual clutter.
- Only nodes where the output *semantics* change (noises, blend) need it.

**3. The current dropdown is a lie**
- Switching Perlin from `Auto` → `UV` only changes the **VkFormat allocation**
  (`R16_SFLOAT` → `R16G16_SFLOAT`). The shader still computes the same
  `vec4(noise, grad.x, grad.y, 1)` — it just wastes the extra channels.
- The REAL potential: format should change what the shader **computes**.

---

## Future Vision: Two Kinds of Format Enforcement

### Category A: Fixed-format nodes (NO dropdown)

These nodes have a **fixed output format** determined by their purpose.
The JSON manifest declares it, the user cannot override it.

| Node | Output Format | Why fixed |
|------|---------------|-----------|
| `image` | RGBA | Input is always a full-color image |
| `color_const` | RGBA | Always produces an RGBA constant |
| `split_rgba` R output | Mono | Conceptually a single channel |
| `split_rgba` G output | Mono | Conceptually a single channel |
| `split_rgba` B output | Mono | Conceptually a single channel |
| `split_rgba` A output | Mono | Conceptually a single channel |
| `combine_rgba` output | RGBA | Produces a full 4-channel image |
| `invert` | Inherit from input | The format is whatever the input is |
| `grayscale` | Mono | Always produces mono output |
| `output` | RGBA | Always full color (blitted to display) |

**Implementation:**

The JSON manifest declares `"format"` per output socket. When `format` is
present, `factory.py` should set `format_override` to that value and
**not expose it in draw_buttons**. The `TextureSynthNode` base class
should have `format_override` visible only when a class attr
`_show_format_override = True`.

```python
class TextureSynthNode:
    _show_format_override: bool = False  # default: hidden

    def draw_buttons(self, context, layout):
        if self._show_format_override:
            layout.prop(self, 'format_override', text="")
```

### Category B: Format-variant nodes (SHOW dropdown)

These nodes change their **mathematical behavior** based on format.

| Node | Available Formats | What changes |
|------|------------------|--------------|
| `perlin` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `simplex` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `worley` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `gabor` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `value` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `white_noise` | Auto, Mono, UV, RGB, RGBA | Shader variant per format |
| `blend` | Auto (inherit from inputs) | Must match input B's format |

---

## The Big Idea: Shader Variants Per Format

The current `ShaderVariantKey` system (`src/engine/ShaderVariantKey.hpp`)
already has a `feature_flags` field for this:

```cpp
struct ShaderVariantKey {
    std::string node_type_id;
    uint32_t    input_count      = 0;
    uint32_t    param_socket_mask = 0;
    uint32_t    feature_flags     = 0;  // ← unused today
};
```

The plan: **populate `feature_flags` from the selected format**.

| Format | Flag | Shader variant |
|--------|------|----------------|
| Mono   | 0    | Returns `vec4(noise, noise, noise, 1)` |
| UV     | 1    | Returns `vec4(grad.x * 0.5 + 0.5, grad.y * 0.5 + 0.5, 0, 1)` |
| RGB    | 2    | Returns `vec4(noise, grad.x * 0.5 + 0.5, grad.y * 0.5 + 0.5, 1)` |
| RGBA   | 3    | Returns `vec4(noise, grad.x * 0.5 + 0.5, grad.y * 0.5 + 0.5, 1)` (same as RGB with alpha) |

### How it flows:

```
1. User selects "UV" on Perlin node
       │
       ▼
2. Python: format_override = 'UV'
       │
       ▼
3. Python: map 'UV' → format_flag = 1
       │
       ▼
4. Python: graph.add_node(id, "perlin", ChannelFormat.UV)
       │
       ▼
5. C++ ResourceManager: allocate R16G16_SFLOAT image (correct VkFormat)
       │
       ▼
6. C++ ShaderCache: key = ("perlin", input_count=0, param_socket_mask=0, feature_flags=1)
       │
       ▼
7. GraphCompiler: emit different GLSL body based on feature_flags
       If flag == 0 (Mono):
           result = vec4(noise, noise, noise, 1)
       If flag == 1 (UV):
           result = vec4(noise_uv.x, noise_uv.y, 0, 1)  ← DIFFERENT MATH
       If flag == 2 (RGB):
           result = vec4(noise, noise_uv.x, noise_uv.y, 1)
```

### The GLSL node function needs to output more data

To support this, noise nodes need to compute extra values internally.
For example, `perlin.glsl` already returns `vec4(noise, ∂n/∂x, ∂n/∂y, 1)`.
This means UV and RGB variants are essentially **reinterpretations of
existing computed data** — no extra performance cost.

A node that does NOT compute derivatives (e.g., `value` noise) would
need a different approach for UV:
- For `value` noise with UV format, the node could be modified to also
  compute gradients numerically (small extra cost) or just return
  the raw position as UV.

### Node manifest declares variant support

The node JSON should declare which formats it supports:

```json
{
    "id": "perlin",
    "variant_flags": ["mono", "uv", "rgb", "rgba"],
    "formats_supported": ["MONO", "UV", "RGB", "RGBA"]
}
```

This lets:
1. The Python UI show only valid formats for each node
2. The GraphCompiler know which variants to special-case
3. The ShaderCache key correctly on the variant

---

## Implementation Steps

### Phase 1: Remove format_override from fixed-format nodes

**Goal:** Only show dropdown on nodes where it makes sense.

1. Add class attribute `_show_format_override = False` to `TextureSynthNode`
2. Override `= True` in specialized nodes that need it (`blend.py`)
3. For auto-generated nodes: set `_show_format_override = True` only if
   the node's JSON declares `"formats_supported"` with >1 entry
4. In `draw_buttons`, gate `layout.prop(self, 'format_override')` behind
   `self._show_format_override`
5. Fixed-format nodes still call `self.rebuild_output_sockets()` during
   `init()`, but the dropdown is invisible

### Phase 2: Shader variant support per format

**Goal:** Format selection changes the compiled shader variant, not just
  the VkFormat.

1. Add `format_flag` to `ShaderVariantKey` (use existing `feature_flags`)
2. `GraphCompiler::compile()` reads the node's output format and passes
   it to `build_variant_key()`
3. `emit_node_shader()` switches on the flag:
   - **flag 0** (Mono): default behavior (current)
   - **flag 1** (UV): emit derivative-based output
   - **flag 2** (RGB): emit noise + 2-channel extra
   - **flag 3** (RGBA): emit full 4-channel
4. `ShaderCache` automatically handles different SPIR-V per variant

### Phase 3: Per-output format in JSON manifest

**Goal:** The manifest declares the intent; the engine enforces it.

Update the node JSON schema:

```json
{
    "outputs": [
        {
            "name": "R",
            "type": "vec4",
            "format": "mono",
            "format_variants": []  ← empty = fixed, never overridden
        }
    ],
    "formats_supported": ["MONO", "UV", "RGB", "RGBA"]
}
```

### Phase 4: Python-side matching

**Goal:** The dropdown only shows formats the node advertises.

In `factory.py`:

```python
def _format_override_items_for_node(node_type):
    """Return FORMAT_OVERRIDE_ITEMS filtered to what this node supports."""
    supported = getattr(node_type, 'formats_supported', None)
    if not supported or len(supported) <= 1:
        return []  # no dropdown at all
    return [
        item for item in FORMAT_OVERRIDE_ITEMS
        if item[0] in supported or item[0] == 'DEFAULT'
    ]
```

And pass the filtered items to `EnumProperty(items=...)` when
constructing the node class dict.

---

## SVG Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER PICKS FORMAT ON A NODE                       │
│                                                                     │
│    e.g. Perlin noise → format = "UV"                                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BLENDER UI (factory.py)                                            │
│                                                                     │
│  format_override_items = node_type.formats_supported                │
│  → only show "MONO", "UV", "RGB", "RGBA"                           │
│  → "Auto" defaults to JSON-declared format                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENGINE BRIDGE (engine_bridge.py)                                   │
│                                                                     │
│  graph.add_node(id, "perlin", ChannelFormat.UV)                     │
│  → C++ stores format_override on NodeInstance                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────────┐  ┌───────────────────────────────┐
│ ResourceManager       │  │ GraphCompiler                 │
│                       │  │                               │
│ Uses format_override  │  │ Reads format_override from    │
│ to pick VkFormat:     │  │ NodeInstance → converts to    │
│                       │  │ feature_flags bitmask:        │
│ UV → R16G16_SFLOAT    │  │                               │
│ Mono → R16_SFLOAT     │  │ flag=0 (Mono): default body   │
│ RGBA → R16G16B16A16   │  │ flag=1 (UV): emit UV-deriv    │
│                       │  │ flag=2 (RGB): emit RGB body   │
│  (already works!)     │  │ flag=3 (RGBA): full color     │
└───────────────────────┘  └───────────┬───────────────────┘
                                       │
                                       ▼
                      ┌───────────────────────────────┐
                      │ ShaderCache                   │
                      │                                │
                      │ Key = (type_id, input_count,   │
                      │        param_socket_mask,      │
                      │        feature_flags)          │
                      │                                │
                      │ → Mono Perlin and UV Perlin    │
                      │   are DIFFERENT cached shaders │
                      └───────────────────────────────┘
```

---

## Open Questions

1. **What about nodes with multiple outputs?**
   - `split_rgba` has 4 outputs, each fixed to Mono.
   - A hypothetical "gradient" node might output Mono + UV simultaneously.
   - Decision: each output socket has its own `format` in the JSON manifest.
     Format override applies to ALL outputs of the node (simpler).
     If per-output formatting is needed later, the data model supports it.

2. **Does changing format invalidate downstream nodes?**
   - Yes — different VkFormat means different image allocation.
   - The bridge handles this naturally: `submit_graph()` → `set_graph()`
     → `ResourceManager::allocate_for_graph()` creates new images.
   - Topology change. Dirty propagation picks up everything downstream.

3. **How do we handle "Auto" format?**
   - Auto = use the JSON manifest's declared format.
   - For Perlin: JSON declares `"format": "rgba"`, so Auto = RGBA.
   - The user can override to Mono/UV/RGB/RGBA as needed.

4. **Can a node expose different formats per output socket?**
   - Currently: no. `format_override` applies to all outputs.
   - The JSON manifest already supports per-socket `format`.
   - Future: perhaps each output socket could have its own dropdown, but
     this adds significant UI complexity. Simplified: the current single
     dropdown is fine for all practical use cases.

5. **What about `as_socket` params?**
   - Socket-driven params are inputs, not outputs. Their format is
     determined by the upstream node, not the downstream. No change needed.

---

## Quick Reference: Which-Socket-Can-Connect-To-What

Downstream input \ Upstream output | Mono | UV | RGB | RGBA | Int
---|---|---|---|---|---
**MonoSocket** | ✓ | ✗ | ✗ | ✗ | ✗
**UVSocket** | ✓ (reads .r) | ✓ | ✓ (reads .rg) | ✓ (reads .rg) | ✗
**ColorSocket** | ✓ (reads .r) | ✓ (reads .rg) | ✓ | ✓ | ✗
**IntSocket** | ✗ | ✗ | ✗ | ✗ | ✓

*sverchok uses colored sockets and allows implicit conversions.
We could eventually add socket color → format hints.
But for now, format_override controls the socket color via
`FORMAT_SOCKET_MAP` in `tree.py`.*
