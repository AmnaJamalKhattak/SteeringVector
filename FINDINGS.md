# Steering Vectors for FLUX — Findings & Architecture Notes

## FLUX Architecture: Two Independent Text Pathways

The most critical insight from diagnostic experiments is that FLUX has **two independent text pathways** with fundamentally different roles:

```
CLIP path:  pooled_text → time_text_embed (MLP) → shift/scale MODULATION → every block
T5 path:    T5_embeddings → context_embedder (Linear 4096→3072) → add_k/add_q → joint attention
```

### Diagnostic Zeroing Results

| Test | Result | Implication |
|------|--------|-------------|
| Zero `context_embedder` (T5) | Minimal effect — still produces a dog | T5 path irrelevant for object identity |
| Zero `time_text_embed` (CLIP) | Pure noise — image unrecognizable | CLIP IS the concept source for objects |

**Conclusion:** Object identity (dog, cat, car, etc.) flows through CLIP → modulation. T5 provides style/texture/detail through attention K/Q, but is not the source of object concepts.

---

## Mode History: What We Tried

### Modes that were removed

| Mode | What it steered | Result | Why it failed |
|------|----------------|--------|---------------|
| `entry_point` | context_embedder + time_text_embed | Partial for style | Only 2 control points, too coarse for objects |
| `block` | to_out[0] + proj_out (text slice) | Moderate | SingleStream text slicing added complexity without clear gain |
| `double_proj` | add_k_proj + add_q_proj | Style OK, objects fail | Operates on T5 path — irrelevant for objects |
| `all` | entry_point + double_proj + double to_out | Noisy | Too many surfaces, conflicting steering directions |
| `object` | add_v_proj + to_out[0] + SingleStream image slice | Mixed | Over-engineered, renormalization masked underlying issues |
| `object_v2` | entry points + to_out[0] + SingleStream image | Mixed | Required diverse prompts, still too many surfaces |
| `joint_attn` | to_out[0] only (19 DoubleStream blocks) | **Worked for dog** | Simple and effective, but no CLIP source steering |

### The `pincer_v2` bug (add_k/add_q version)

The first version of `pincer_v2` steered `time_text_embed` + `add_k_proj` + `add_q_proj`. Results showed **97% of pixels changed with mean diff ~38**, but the dog persisted in every image.

**Root cause:** `add_k_proj` / `add_q_proj` operate on T5 context — the pathway that is **irrelevant for object identity**. We were massively disrupting T5-based attention (which changes textures, details, and composition) without ever touching where "dog" actually lives (CLIP → modulation).

---

## Final Architecture: 2 Modes

### Mode 1: `hybrid` — Style Unlearning

**Control points:**
- `context_embedder`: Linear(4096 → 3072) — projects T5 encoder output
- `time_text_embed`: MLP — fuses timestep + CLIP pooled text
- `add_k_proj` + `add_q_proj` in 19 DoubleStream blocks — text-side K/Q

**Why this works for style:** Style IS carried through T5 descriptions (texture, composition, aesthetic cues). The T5 path is the correct target. Entry points provide additional coverage at the source.

**Key details:**
- Mask-aware pooling (excludes T5 padding tokens from mean)
- Keeps ALL vectors (no top_k filtering — style is distributed)
- Single scalar beta (2.0–5.0 typical range)

### Mode 2: `pincer_v2` — Object Unlearning

**Control points:**
- `time_text_embed` (CLIP): Weaken global concept signal at its **source**
- `to_out[0]` in 19 DoubleStream blocks: Steer where concept **manifests** in image representation

**Why this works for objects:**
1. `time_text_embed` is the CLIP source — zeroing it produces noise, confirming it carries object identity
2. `to_out[0]` is the image representation AFTER all processing (CLIP modulation + T5 attention combined) — the closest FLUX analog to SD's cross-attention output that CASteer targets
3. The "pincer" approach targets both supply (CLIP) and usage (to_out) simultaneously

**Key details:**
- Simple mean pooling for both (no T5 mask needed — CLIP is global, to_out is image-stream)
- Top-k=15 for to_out vectors (proven in joint_attn experiments)
- ALL CLIP vectors always kept (just 1 per timestep, already sparse)

**Per-component beta:**
```python
beta = {"clip": 3.0, "attn": 10.0}
```
- `clip` (time_text_embed): Gentle — CLIP is fragile, carries ALL conditioning (not just the target concept)
- `attn` (to_out[0]): Aggressive — more concept-specific, can handle stronger intervention

---

## Key Design Decisions

### 1. Diverse Prompt Pairs (CASteer Methodology)

Single prompt pairs ("Dog" / "Object") capture general content, not the specific concept. This caused full image distortion — the dog was removed but so was everything else.

**Fix:** 50 diverse ImageNet contexts as base prompts:
```
("a photo of a tench with a Dog", "a photo of a tench")
("a photo of a goldfish with a Dog", "a photo of a goldfish")
("a photo of a shark with a Dog", "a photo of a shark")
...
```

The diverse contexts cancel out, leaving only the concept-specific activation direction. This is the CASteer methodology: many pairs × 1 seed > 1 pair × many seeds.

### 2. Per-Component Beta

CLIP and attention pathways have fundamentally different properties:
- **CLIP (time_text_embed):** Global modulation signal. Carries ALL conditioning — not just "dog" but also scene layout, lighting, everything. Aggressive steering here destroys the entire image.
- **to_out[0]:** Image representation after attention. More concept-specific — the "dog-ness" direction is more isolated from general scene content.

The 3:10 ratio (clip:attn) reflects this: gentle at the fragile source, aggressive at the specific manifestation.

### 3. Top-k Selection for to_out

Not all 19 blocks × 4 steps = 76 possible to_out vectors contribute equally. Top-k=15 selects the strongest concept-specific directions, reducing noise. The CLIP vector (1 per step = 4 total) is always kept since it's the source.

### 4. Mask-Aware Pooling (Hybrid Mode Only)

T5 context is padded to max_length=512 tokens. Without masking, mean pooling is dominated by ~500 padding positions, collapsing steering magnitude by ~85x ("Padding-Inclusive GAP" bug from early development).

Formula: `masked_mean = sum(act * mask, dims) / sum(mask, dims)`

This is only needed for the T5 path (hybrid mode). Pincer_v2 doesn't use it since CLIP output is (B, D) with no sequence dimension, and to_out is image-stream.

---

## Steering Formula

All modes use the same projection removal:

```
output' = output - beta * clamp(output · d, min=0) * d
```

Where `d` is the unit-norm concept direction, `beta` controls strength, and `clamp(min=0)` ensures we only remove positive projections onto the concept direction (prevents amplifying negative projections).

Setting `clip_negative=False` removes the clamp and steers both positive and negative projections. Try this if the concept persists — the learned direction might be flipped for certain eval prompts.

---

## Evaluation: UnlearnCanvas Benchmark

### Metrics
- **UA (Unlearning Accuracy):** 1 - classifier accuracy on target concept (higher = better forgetting)
- **IRA (In-domain Retain Accuracy):** Accuracy on same-domain non-target concepts (higher = better preservation)
- **CRA (Cross-domain Retain Accuracy):** Accuracy on other-domain concepts (higher = better preservation)
- **FID:** Frechet Inception Distance for image quality (lower = better)
- **CLIP Score:** Text-image alignment (higher = better)

### Classifier
Primary: LLaVA-1.6-Vicuna-7B following TRACE paper (ICLR 2026) exact prompt format with numbered lists. Fallback: CLIP ViT-L/14 for fast evaluations.

### Evaluation Grid
10 styles × 20 objects × multiple eval seeds. Full pipeline: Generate with steering → Free FLUX VRAM → Classify with LLaVA → Reload FLUX.

---

## Configuration Defaults

```python
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
N_STEPS = 4                          # FLUX.1-schnell inference steps
LEARNING_SEEDS = list(range(0, 20))  # 20 seeds for vector learning
TOP_K_VECTORS = 15                   # Top-k for pincer_v2 to_out selection
NUM_DIVERSE_PROMPTS = 50             # CASteer diverse prompt pairs

# Style unlearning
STEERING_MODE = "hybrid"
BETA = 2.0  # scalar, range 2-5

# Object unlearning
STEERING_MODE = "pincer_v2"
BETA = {"clip": 3.0, "attn": 10.0}  # per-component
```

---

## Summary Table

| Aspect | Style (`hybrid`) | Object (`pincer_v2`) |
|--------|-----------------|---------------------|
| Concept pathway | T5 → attention K/Q | CLIP → modulation |
| Steered layers | context_embedder + time_text_embed + add_k/add_q | time_text_embed + to_out[0] |
| Pooling | Mask-aware (T5 padding) | Simple mean |
| Vector selection | Keep all | CLIP: all, to_out: top_k=15 |
| Beta | Scalar (2–5) | Dict: clip=3, attn=10 |
| Prompt strategy | Diverse style pairs | Diverse object pairs |
| Total control points | 2 + 19 + 19 = 40 | 1 + 19 = 20 |
