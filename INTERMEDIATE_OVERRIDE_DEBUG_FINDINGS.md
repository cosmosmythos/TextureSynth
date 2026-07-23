# Intermediate Override Bug: Visual Trace

## The graph

```
Worley(1) → Levels(2) → Blur(3) → Invert(4) ─┐
                                ┌───────────────┘
                                ↓
Levels(2) → Invert.001(5) → Blur.001(6) ──────┐
                                        ┌──────┘
                                        ↓
Perlin(7) → Levels.001(8) → Blend(9) → Levels.002(10) → Blur.002(11) ──┐
                                                                 ┌──────┘
                                                                 ↓
                                                            Blend.001(12) = output
```

3 blurs, each is actually 2 passes internally (horizontal pass0 → intermediate → vertical pass1).

## How the engine splits this into groups

The fusion pipeline (`FusionGroup.hpp`) walks nodes in order and groups connected nodes. Each blur expands to 2 entries (pass0, pass1). Groups split at Sampler2D connections.

**Result: 7 groups**

```
group 0: [1,2]               pass 0/1   ← Worley + Levels (single pass)
group 1: [5]                 pass 0/1   ← Invert.001
group 2: [6]                 pass 1/2   ← Blur.001 pass1 (NO pass0 group exists!)
group 3: [3]                 pass 0/2   ← Blur pass0 → CREATES intermediate key=3
group 4: [3,4,7,8,9,10]     pass 0/1   ← Blur pass1? NO — this is the merged chain
group 5: [11]                pass 0/2   ← Blur.002 pass0 → CREATES intermediate key=11
group 6: [6,11,12]           pass 1/2   ← TWO blurs pass1 + blend tail
```

## The code that handles pass1 groups

`EngineGroupCompile.cpp`, inside `populate_groups_()`:

```cpp
// Line 137: loop over every group
for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
    const auto& cg = compiled.groups[gi];

    // ... compile GLSL, allocate images ...

    // Line 149: loop over every external Sampler2D input of this group
    for (size_t ei = 0; ei < cg.external_inputs.size(); ++ei) {
        const auto& ext = cg.external_inputs[ei];

        // Line 163: THE BUGGY LINE — only fires for ei==0
        if (cg.pass_count > 1 && cg.pass_index == 1 && ei == 0) {
            auto inter_it = intermediates.find(cg.output_node);
            //                        ^^^^^^^^^^^^^^^^
            //                        ALSO BUGGY — looks up the GROUP TAIL node,
            //                        not the actual blur node
            if (inter_it != intermediates.end()) {
                // redirect Sampler2D to intermediate image
                input.sampled_slot = inter_it->second.sampled_slot;
                continue;
            }
        }
        // ... falls through here if override missed or ei != 0
        // blur reads from the ORIGINAL source image instead of intermediate
```

## Trace each pass1 group

### Group 2: `[6] pass=1/2`

```
intermediates at this point = {} (empty — nothing created pass0 yet)

cg.output_node = 6    ← the only node, so tail = blur.001(6)
ei=0: ext.dst_node = 6, ext.src_node = 5

→ enters override (pass_count>1, pass_index==1, ei==0)
→ intermediates.find(6) where intermediates={}
→ NOT FOUND! available=[]
→ falls through → blur.001 reads from src_node=5 (original image)
```

**Why it fails:** Group 2 runs BEFORE group 3 (which creates the first intermediate). Blur.001's pass0 group was never created — it was merged into another group during fusion and lost its multi-pass status.

### Group 6: `[6,11,12] pass=1/2`

```
intermediates at this point = {3: inter, 11: inter}
                                ↑ key=3 (blur)  ↑ key=11 (blur.002)
                                NOT key=12 (blend tail)

cg.output_node = 12   ← group tail = blend.001, NOT the blur nodes!

ei=0: ext.dst_node=6  (blur.001's Sampler2D)
→ enters override (pass_count>1, pass_index==1, ei==0)
→ intermediates.find(12)  ← searches for blend tail node
→ NOT FOUND! available=[11,3]
→ falls through → blur.001 reads from src_node=5 (original image)

ei=1: ext.dst_node=11 (blur.002's Sampler2D)
→ ei==0 is FALSE → override NEVER fires
→ falls through → blur.002 reads from src_node=10 (original image)
```

**Two failures in one group:**
1. `intermediates.find(12)` searches by blend tail instead of blur node
2. `ei==0` only checks the first input; blur.002 at `ei=1` is invisible to the override

## What `intermediates` actually contains

```
After all groups processed:

intermediates = {
    3:  {image_for_blur_pass0},    ← created by group 3
    11: {image_for_blur002_pass0}  ← created by group 5
}
```

These are keyed by the **blur node ID** because that's what each pass0 group's `output_node` was.

## But the code searches by group tail

```cpp
intermediates.find(cg.output_node);
```

For group 6 (`[6,11,12]`), that's `cg.output_node = 12` — the blend. `intermediates[12]` doesn't exist. The correct keys to search for are `6` and `11` — but those live in `ext.dst_node`, not in `cg.output_node`.

## The fix in one sentence

```cpp
// Line 163-164 — replace:
if (cg.pass_count > 1 && cg.pass_index == 1 && ei == 0) {
    auto inter_it = intermediates.find(cg.output_node);

// With:
if (cg.pass_count > 1 && cg.pass_index == 1) {
    auto inter_it = intermediates.find(ext.dst_node);
```

Check every external input (remove `ei==0`), and look up by the input's target node (`ext.dst_node`) instead of the group tail (`cg.output_node`).
