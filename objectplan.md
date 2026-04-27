# Plan: 2-vector object unlearning in styleunlearnflux.py

## Context

Goal: a method in `styleunlearnflux.py` that unlearns *objects* with the same architectural footprint that already works for *styles* — exactly **2 steered vectors**, intervention right at the two text entry points (CLIP `time_text_embed` input, T5 `context_embedder` input), and minimal change to non-object content.

What we tried so far that failed (pincer_v2 SVD auto-rank, sub-class diversification of `make_object_prompts`, pincer_perstep with 5 vectors). Each new run still left the dog visibly intact (diff_mean ≈ 28.5, 94% pixels changed but object preserved). The latest pincer_perstep run produced "Vectors: 5 total" but identical numbers — confirming the failure is **not** at the vector-extraction layer.

A focused mechanical comparison with the working `objectunlearnflux.py` showed the actual cause is in the **apply** path of `styleunlearnflux.py`, which silently throttles every object-mode run:

| Knob | styleunlearnflux (object run) | objectunlearnflux (working) |
|---|---|---|
| CLIP cap in `clip_pre_hook` | `min(b_clip, 1.0)` (line 1264) — limits to 1× the projection | none — `b_clip * score` directly |
| Object `BETA["clip"]` default | 1.0 (cap-bound) | 3.0 |
| Object `STEP_RANGE` default | `(0, 2)` — silently disables 2 of 4 per-step vectors | full coverage |
| Object `TOP_FRAC` default | 0.15 — gates 85% of tokens | 1.0 in working configs |

The diagnostic at the top of the file (lines 240-243) is the architectural anchor:
- *Zeroing time_text_embed (CLIP) → pure noise → CLIP IS the concept source*
- *Zeroing context_embedder (T5) → still a dog → T5 is irrelevant for object identity*

So CLIP is the lever, and capping it at 1× of its own projection is precisely what prevents the dog from being removed: subtracting `1·score·direction` zeroes one axis of pooled CLIP, but `dog` lives in a multi-dimensional subspace, so the model just slides to a neighbouring breed (which is what every failed image showed). objectunlearnflux works *not* because of its 5 vectors but because at β_clip=3 uncapped, the CLIP path is pushed past zero into anti-dog space; the 4 per-step T5 vectors are along for the ride.

So the intended outcome of this plan: keep pincer_v2's 2-vector architecture, fix the apply path so the same 2 vectors actually do the job for objects, and keep style behavior unchanged.

## Recommended approach

Three small, surgical changes:

### 1. Make the CLIP cap a parameter, not a hard-coded `min(., 1.0)`

`styleunlearnflux.py:1251-1267` — `clip_pre_hook` inside `apply_vectors`.

Add a `clip_cap` parameter to `apply_vectors` and `generate` (default `1.0`, matching the current style behavior). Replace the hard-coded `min(float(b_clip), 1.0) * score` with:

```python
if clip_cap is None:
    effective = float(b_clip) * score
else:
    effective = min(float(b_clip), float(clip_cap)) * score
```

Default `clip_cap=1.0` preserves style behavior exactly. Object mode passes `clip_cap=None` (uncapped).

Reason this is safe for style: style configs use `BETA["clip"]=0.0` (no CLIP component anyway, per the existing comment block at lines 2247-2251). The cap was only ever there to prevent runaway CLIP for objects — and that's exactly what we want for objects now: a *parameter*, not a hard ceiling.

### 2. Update object-mode defaults to the working values

`styleunlearnflux.py:2262-2269` — the `if TARGET_TYPE == "style": ... else: ...` block.

Change the object branch to:

```python
BETA       = {"clip": 3.0, "t5": 4.0}   # was {"clip": 1.0, "t5": 4.0}
TOP_FRAC   = 0.15                        # unchanged — keeps T5 spatially localized to concept tokens
STEP_RANGE = (0, N_STEPS)                # was (0, 2) — fire every step
CLIP_CAP   = None                        # new — uncapped CLIP for objects
```

Reason for each:
- `clip=3.0, clip_cap=None`: matches objectunlearnflux's working config; pushes the CLIP projection past zero so the multi-dimensional `dog` subspace is cleared along the dominant axis.
- `t5=4.0`: existing T5 strength. T5 isn't the lever per the diagnostic, but mild T5 ablation prevents joint attention from "re-encoding" the concept after CLIP suppression.
- `top_frac=0.15`: kept for **selectivity** — limits T5 subtraction to the top-15% tokens by alignment score (i.e., the actual concept tokens), so non-object tokens in the prompt are untouched. This is the user's "minimal changes to background" lever.
- `step_range=(0, N_STEPS)`: required so all 4 denoising steps see the steering. The (0,2) default was silently disabling half the trajectory.

Style branch is unchanged.

### 3. Plumb `clip_cap` through `generate` and the test cell

`styleunlearnflux.py` — `generate(...)` signature (search for `def generate`) and the test cell that calls it (search for `steerer.generate(`). Add `clip_cap=CLIP_CAP` as a keyword argument so the per-target default flows through.

## Side note: revert the dead-end additions (optional but recommended)

The earlier attempts didn't move the needle and add complexity:
- SVD auto-rank + null model in `learn_vectors_diverse` (`styleunlearnflux.py:673-859`) — produced k=1 anyway, with a weaker direction.
- `OBJECT_SUBCLASSES` + `make_object_prompts` sub-class diversification (`styleunlearnflux.py:157-285`) — diluted the per-pair concept signal.
- `pincer_perstep` mode + `_learn_pincer_perstep` (`styleunlearnflux.py:VALID_MODES`, around lines 668-770, plus dispatcher and init summary) — produces 5 vectors which the user explicitly does not want.

Recommend reverting all three so the file goes back to the canonical 2-vector pincer_v2 path; the cap-lift change is what actually fixes the problem.

If you'd rather keep them as escape hatches, that's also fine — they're inert when not selected. Tell me which.

## Critical files & lines to edit

- `styleunlearnflux.py:1251-1267` — `clip_pre_hook` (introduce `clip_cap` param, change the cap line).
- `styleunlearnflux.py:1186-1199` — `apply_vectors` signature (add `clip_cap=1.0`).
- `styleunlearnflux.py:1126-1135` (the `def generate(self, ...)` signature) — add `clip_cap=1.0` and forward it.
- `styleunlearnflux.py:2262-2269` — object-mode defaults (BETA["clip"]=3.0, STEP_RANGE=(0,N_STEPS), add CLIP_CAP=None).
- Test cell that calls `steerer.generate(...)` — pass `clip_cap=CLIP_CAP`. (Verify location during implementation; there are multiple test cells.)
- (Optional revert): `styleunlearnflux.py:157-285` `OBJECT_SUBCLASSES` + `make_object_prompts`; `styleunlearnflux.py:673-859` `learn_vectors_diverse` SVD/null-model branch; `styleunlearnflux.py:VALID_MODES` and `_learn_pincer_perstep` and the dispatcher + init summary entries for pincer_perstep.

## Existing utilities reused (no new code needed)

- `apply_vectors`'s existing `_apply_topk_gate` + `_in_range(step)` provide the selectivity (`top_frac` for tokens, `step_range` for steps) — both already wired and correct.
- `_learn_pincer_v2(self, pos_prompt, neg_prompt, seeds, verbose)` (`styleunlearnflux.py:614-667`) already extracts the canonical 2 vectors (`clip_768` 768-d, `t5_4096` 4096-d). No change needed there — the bug was downstream in `apply`.
- `output_hook(..., token_gate=True)` already handles top-k gating for the T5 path.

## Verification

1. **Style sanity check (must not regress).**
   - `TARGET_CONCEPT="Van_Gogh"`, `TARGET_TYPE="style"`. Confirm the style sweep panel still shows clean style removal at `BETA={"clip":0, "t5":2}`, `clip_cap=1.0` (default). Same images as before this change.

2. **Object end-to-end test.**
   - `TARGET_CONCEPT="Dog"`, `TARGET_TYPE="object"`. Title bar should read "Vectors: 2 total".
   - Expect: `TEST 0a (T5=0)` panel shows much stronger CLIP-only ablation than before — dog visibly broken, not just a different breed (since β_clip=3 uncapped pushes well past the concept-zero point).
   - Expect: param-sweep panels (`top=0.15 steps=(0,4)` etc.) show the dog *gone* — empty park / replaced with non-dog content (matching objectunlearnflux's working behavior visually).
   - Expect: background structure (trees, grass, lighting) recognizably preserved because (a) `top_frac=0.15` keeps T5 steering on concept tokens only and (b) CLIP modulation is global but scene tokens dominate after dog-removal.

3. **Print-out check during learning.**
   - Per-pair print should still log "Total vectors: 2 (clip_768 + t5_4096)" for object mode (confirms we are *not* on the 5-vector path).

4. **Spectrum/strength sanity (verbose).**
   - With β_clip=3 uncapped, the projection score for a dog prompt at runtime should produce a substantial subtraction; if you log `effective.mean()` inside `clip_pre_hook` once for a steered run, expect it to be 3× larger than in the previous capped run.

5. **Side-effect bound.**
   - Compare object panels' diff_mean against baseline. With `top_frac=0.15` + `step_range=(0,N_STEPS)` + uncapped CLIP β=3, expect a high `diff_mean` localized to the dog region and modest changes elsewhere — i.e., the user's "minimal changes to background" goal. If background drift is too large, the next dial is to lower β_clip to 2 or apply `clip_cap=2.0` instead of `None`.

## Out of scope / open question

If, after this change, the dog is only *partially* removed (still visible at lower confidence), the next escalation is rank-2 CLIP — register a second pre-hook on `time_text_embed` for the second SVD component of the same diff matrix. That keeps the "2 entry points" intervention surface but bumps total directions to 3. We would only consider this if uncapped β_clip=3 is insufficient. The current plan does not include this; flag it if the verification step 2 still leaves the dog in place.