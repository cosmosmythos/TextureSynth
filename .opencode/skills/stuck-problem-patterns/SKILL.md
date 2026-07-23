---
name: stuck-problem-patterns
description: Break out of a stuck loop when the user has rejected solutions 2+ times by recognizing when you're optimizing within the wrong mental model.
compatibility: opencode
---

## What I do

When the user rejects solutions multiple times, stop optimizing within your current mental model and question the model itself. You adopt a framework early, then spend all effort tweaking it — but the user sees wrong output while you see a bug to fix. These are different diagnoses.

## How I break the loop

### Step 1: Recognize the loop

Signs you're stuck:
- Your "fixes" are all variations of the same idea
- You can explain why each fix is "almost right"
- The user keeps saying variations of "that's not what I meant"
- You're adding complexity (extra steps, parameters, edge cases) to fix what should be simple

**If you see 2+ of these, stop writing code immediately.**

### Step 2: Write down your current model

One sentence. What are you actually doing? Look for the pattern across all attempts, not the surface details.

### Step 3: Write what the user described

Their exact words. Resist translating into your technical framework.

### Step 4: Ask the exit question

**"What would the simplest implementation look like if I ignored my current code entirely?"**

Write pseudocode. If you can't make it simple, you don't understand the problem yet.

### Step 5: The one-question test

Before any fix: **"If I deleted all my code and started from the user's description alone, would I write this?"**

- No → you're polishing the wrong approach. Go back to Step 2.
- Yes → proceed with the fix.

## When to use me

Use this skill when:
- The user has rejected your solution 2+ times
- You keep adjusting parameters or code without satisfying the user
- You feel "almost there" but the output keeps being wrong
- You catch yourself refining the same approach instead of trying a different one

## What I never do

- Keep polishing the same mental model after repeated rejection
- Treat user feedback as "bugs to fix" instead of "wrong direction"
- Add more complexity to fix what should be simple
- Confuse mathematical correctness with user satisfaction
