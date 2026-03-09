# CTMY Project Results Report: Steering Vectors for Concept Removal in FLUX Diffusion Models

## 1. Introduction

This report documents the methodology, implementation, and experimental results of applying **activation steering vectors** to the **FLUX.1-schnell** diffusion model for targeted concept removal (unlearning). The work is evaluated on the **UnlearnCanvas benchmark** and compared against recent baselines from the TRACE paper (ICLR 2026).

The primary implementation resides in a single notebook:

| Notebook | Role |
|---|---|
| `SteeringVectorsUnlearnCanvasPincerV2.ipynb` | Complete pipeline with 2 steering modes (`hybrid` for styles, `pincer_v2` for objects), per-component beta, diverse CASteer prompt pairs, LLaVA/CLIP classification, FID/CLIP quality metrics, and full UnlearnCanvas benchmark evaluation. |

---

## 2. Methodology: Steering Vectors

### 2.1 Core Idea

Steering vectors are computed as the **difference in internal activations** when the model processes a *positive* prompt (containing the target concept) versus a *negative* prompt (without it). During generation, these vectors are subtracted from the model's activations to suppress the target concept -- all at inference time, without any model fine-tuning or retraining.

### 2.2 Key Architectural Insight

Diagnostic zeroing experiments (Cell 8B) revealed a critical finding about how concept identity flows through FLUX:

- **CLIP path**: `pooled_text -> time_text_embed -> shift/scale modulation -> every block`. Zeroing `time_text_embed` produces **pure noise** -- proving CLIP is the primary concept source.
- **T5 path**: `T5_embeddings -> context_embedder -> add_k/add_q -> joint attention`. Zeroing `context_embedder` still produces recognisable objects -- proving the T5 path is irrelevant for object identity.

This means `add_k_proj` / `add_q_proj` operate on the T5 context (the irrelevant path for objects). Object identity flows through **CLIP -> modulation**, not through T5 -> attention K/Q. This insight is the foundation for the `pincer_v2` mode design.

### 2.3 Computation of Steering Vectors

**Step 1 -- Hooking target layers.** Forward hooks are registered on specific layers inside the FLUX transformer. The exact layers depend on the steering mode (see Section 2.5).

**Step 2 -- Collecting activations.** Using the **diverse CASteer methodology** (50 ImageNet classes as base contexts):
- For each (positive, negative) prompt pair (e.g., "tench with Dog" / "tench"), run a forward pass with both prompts using the same seed.
- Record activations at each hooked layer and each denoising step.
- Averaging across 50 diverse contexts cancels out context-specific noise, isolating the pure concept direction.

**Step 3 -- Averaging differences.** The mean activation difference is computed per layer per denoising step:

```
diff[layer][step] = mean( activation_positive ) - mean( activation_negative )
```

Each difference vector is then L2-normalised to unit norm.

**Step 4 -- Top-k selection (pincer_v2 only).** For `to_out[0]` vectors, candidates are ranked by gradient norm (magnitude of the raw difference before normalisation). Only the top-k most impactful (layer, step) pairs are retained. CLIP vectors (`time_text_embed`) are always kept.

### 2.4 Application of Steering Vectors (Inference-Time)

During image generation, the learned vectors are applied via forward hooks. At each denoising step, for each hooked layer:

1. Compute the **projection score**: `score = output @ target_direction` (dot product along the steering direction).
2. Optionally **clip** negative scores to zero (`clip_negative` mode) -- ensures steering only fires when the activation is aligned with the concept direction.
3. Compute the **update**: `update = (beta * score).unsqueeze(-1) * target_direction`.
4. **Subtract** the update from the layer output: `output = output - update`.

The steering strength is controlled by the hyperparameter **beta (β)**. In `pincer_v2` mode, beta is specified **per-component**:
- `beta["clip"]` -- gentle steering for `time_text_embed` (CLIP is fragile; carries ALL conditioning)
- `beta["attn"]` -- aggressive steering for `to_out[0]` (more concept-specific, can tolerate higher values)

### 2.5 Steering Modes

The notebook implements 2 refined steering modes, each targeting different concept types:

| Mode | Layers Targeted | Best For | Typical Beta |
|---|---|---|---|
| **`hybrid`** | `context_embedder` (T5->model) + `time_text_embed` (CLIP+timestep MLP) + `add_k_proj`/`add_q_proj` in 19 DoubleStream blocks | **Style unlearning** | β = 2.0-5.0 (scalar) |
| **`pincer_v2`** | `time_text_embed` (CLIP source) + `to_out[0]` in 19 DoubleStream blocks (image attention output) | **Object unlearning** | β = {"clip": 3.0, "attn": 10.0} |

#### Mode 1: `hybrid` (Style Unlearning)

Combines TRACE entry points with CASteer-adapted text-side projections:
- **`context_embedder`**: Linear(4096 -> 3072) -- projects T5 encoder output into the transformer.
- **`time_text_embed`**: MLP -- fuses timestep information with CLIP pooled text representation.
- **`add_k_proj` + `add_q_proj`**: Text-side Key and Query projections in all 19 DoubleStream blocks.

Uses **mask-aware pooling** for T5 representations (masking out padding tokens via the T5 attention mask). Keeps **all** learned vectors (no top-k filtering). Style IS carried through T5 descriptions, so steering add_k/add_q is correct here.

#### Mode 2: `pincer_v2` (Object Unlearning)

A two-pronged approach targeting the concept at its source and where it manifests:

- **CLIP source (`time_text_embed`)**: Weakens the global concept signal at its origin. Requires gentle beta since CLIP carries all conditioning.
- **Image attention output (`to_out[0]`)**: Steers where the concept manifests in image representations AFTER all processing (modulation + attention combined). This is the closest FLUX analogue to SD's cross-attention output that CASteer targets. The `to_out[0]` captures the combined downstream effect of both CLIP modulation and T5 attention, so subtracting the concept direction here is effective.

Uses **top-k selection** for `to_out` vectors (default k=15, proven effective in earlier joint_attn experiments). CLIP vectors are always retained.

### 2.6 Target Model

- **Model**: `black-forest-labs/FLUX.1-schnell` (distilled variant)
- **Denoising steps**: 4 (Schnell)
- **Precision**: bfloat16
- **Architecture**: FLUX transformer with 19 double-stream blocks (`FluxTransformerBlock`) and 38 single-stream blocks (`FluxSingleTransformerBlock`)

---

## 3. Experiments and Results

### 3.1 Evaluation Protocol: UnlearnCanvas Benchmark

The evaluation follows the UnlearnCanvas benchmark protocol, adapted for FLUX:

- **10 artistic styles** (from TRACE paper Section 5.1): Van_Gogh, Watercolor, Cartoon, Cubism, Winter, Pop_Art, Ukiyoe, Impressionism, Byzantine, Bricks
- **20 object classes** (singular form, matching TRACE prompts): Architecture, Bear, Bird, Butterfly, Cat, Dog, Fish, Flame, Flowers, Frog, Horse, Human, Jellyfish, Rabbits, Sandwich, Sea, Statue, Tower, Tree, Waterfalls
- **Prompt format**: `"A {Object} image in {Style} style."`
- **Learning seeds**: 20 seeds (range 0-19) for vector computation
- **Evaluation seeds**: 3 seeds (range 20-22) per style-object combination
- **Total images per target concept**: 10 styles x 20 objects x 3 seeds = 600 images

### 3.2 Classification Method

**LLaVA-1.6-Vicuna-7B** is the primary classifier, following the methodology from the TRACE paper (Appendix E.4). The TRACE paper demonstrates that UnlearnCanvas's original SD1.5-trained classifier generalises poorly to modern models like FLUX (<6% accuracy). LLaVA provides accurate zero-shot classification using a numbered-list multiple-choice prompt format.

**CLIP ViT-L/14** is available as a faster alternative (~0.1s/image vs ~2-5s/image for LLaVA), configurable via `USE_LLAVA = False`.

### 3.3 Quantitative Metrics

Three core metrics from the UnlearnCanvas benchmark:

| Metric | Definition | Goal |
|---|---|---|
| **UA (Unlearning Accuracy)** | `1 - (images classified as target / total target-concept images)` | Higher is better (target concept is suppressed) |
| **IRA (In-Domain Retain Accuracy)** | Classification accuracy on other concepts in the **same domain** (e.g., other styles when unlearning a style) | Higher is better (related knowledge preserved) |
| **CRA (Cross-Domain Retain Accuracy)** | Classification accuracy on concepts in the **other domain** (e.g., object accuracy when unlearning a style) | Higher is better (unrelated knowledge preserved) |

Two additional quality metrics:

| Metric | Tool | Goal |
|---|---|---|
| **FID (Frechet Inception Distance)** | `clean-fid` library | Lower is better (image quality/distribution match) |
| **CLIP Score** | CLIP ViT-L/14 cosine similarity | Higher is better (text-image alignment) |

### 3.4 Pipeline Workflow

The notebook follows a two-phase approach for VRAM efficiency:

**Phase 1 -- Generation**: Load FLUX, learn steering vectors, generate all benchmark images (with steering) and baseline images (without steering), save to disk.

**Phase 2 -- Classification**: Unload FLUX from GPU, load LLaVA classifier, classify all saved images, compute UA/IRA/CRA metrics. Resume support is built in (existing images are skipped on re-run).

The full benchmark mode (`RUN_FULL_BENCHMARK = True`) loops over all 10 styles, learning vectors and evaluating each in sequence with incremental CSV output.

### 3.5 Qualitative Results from Diagnostic Tests

The diagnostic test (Cell 8B) generates an 8-panel comparison grid for each target concept. Based on the result images:

#### Cat Unlearning (`pincer_v2`)

| Test | Observation |
|---|---|
| T5 zeroed (context_embedder=0) | Minimal change -- cat still clearly visible, confirming T5 is NOT the object identity source |
| CLIP zeroed (time_text_embed=0) | Pure noise/static -- confirming CLIP is the primary concept source |
| Baseline (no steering) | Clear cat in kitchen scene |
| clip=2, attn=5 | Cat begins to fade; some morphological distortion visible |
| clip=3, attn=10 (default) | Cat significantly suppressed; replaced by amorphous shapes or absent entirely |
| clip=5, attn=15 | Strong removal; image degrades toward noise at highest settings |
| clip=5, attn=20 | Near-complete removal but scene coherence also affected |

Additional beta sweep experiments show:
- **Beta 5/40 (clip/attn)**: Cat replaced by small cat-like artifact, scene mostly preserved
- **Beta 10/40**: Cat fully removed, minor scene artifacts
- **Beta 20/40**: Cat fully removed, scene begins to wash out
- **Beta 40**: Strong removal but noticeable scene degradation
- **Beta 50**: Excessive -- horizontal banding artifacts appear

#### Dog Unlearning (`pincer_v2`)

| Test | Observation |
|---|---|
| T5 zeroed | Dog still present (confirms T5 irrelevance for objects) |
| CLIP zeroed | Pure noise (confirms CLIP is source) |
| Baseline | Clear dog in park/kitchen scenes |
| clip=3, attn=10 (default) | Dog significantly suppressed; replaced by scene-appropriate content |
| No attn steering (CLIP only) | Dogs still partially visible -- CLIP steering alone insufficient |
| High attn steering | Dogs fully removed; scenes remain coherent |
| pincerV2 latest (19 vectors) | Clean removal with good scene preservation |

The ablation between "no attn steering" and "high attn steering" demonstrates that both components of the pincer are necessary: CLIP weakens the global signal, while `to_out[0]` removes the concept where it has already manifested in image representations.

#### Surgical Unlearning Comparison (Cat)

The `to_v.png` result compares three steering strategies for cat removal:
- **Subtract**: Removes cat but produces line-drawing artifacts
- **Orthogonal**: Produces near-complete noise
- **Block targeting**: Preserves scene but cat remains partially visible

This motivated the final `pincer_v2` design which combines source-level (CLIP) and output-level (`to_out`) steering.

### 3.6 Comparison with Baselines

The notebook includes comparison tables against published FLUX baselines from the TRACE paper (Table 1) for **style removal**:

| Method | UA (%) | IRA (%) | CRA (%) | FID |
|---|---|---|---|---|
| LOCOEDIT (FLUX) | 66.45 | 33.23 | 83.44 | 55.56 |
| UCE (FLUX) | 67.43 | 34.78 | 76.56 | 58.90 |
| TRACE (FLUX) | 88.60 | 36.10 | 96.40 | 51.67 |
| **Ours (Steering)** | **Evaluated per run** | **Evaluated per run** | **Evaluated per run** | **Evaluated per run** |

For broader context, SD1.5 baselines from TRACE Table 2 are also included:

| Method | UA (%) | IRA (%) | CRA (%) |
|---|---|---|---|
| ESD | 98.58 | 80.97 | 93.96 |
| FMN | 88.48 | 56.77 | 46.60 |
| UCE | 98.40 | 60.22 | 47.71 |
| CA | 60.82 | 96.01 | 92.70 |
| SalUn | 86.26 | 90.39 | 95.08 |
| SEOT | 56.90 | 94.68 | 84.31 |
| SPM | 60.94 | 92.39 | 84.33 |
| EDiff | 92.42 | 73.91 | 98.93 |
| SHS | 95.84 | 80.42 | 43.27 |
| SAeUron | 95.80 | 99.10 | 99.40 |
| TRACE | 95.02 | 93.84 | 86.22 |

Note: SD1.5 numbers are NOT directly comparable to FLUX results and are provided for broader context only.

### 3.7 Hyperparameter Guidance

Based on diagnostic experiments:

| Concept Type | Mode | Beta | Top-k | Notes |
|---|---|---|---|---|
| Style | `hybrid` | 2.0-5.0 (scalar) | All kept | T5 path matters for style descriptions |
| Object | `pincer_v2` | {"clip": 3.0, "attn": 10.0} | 15 | Per-component beta critical; CLIP is fragile |

The per-component beta is essential for `pincer_v2`: CLIP embedding carries ALL conditioning (not just the target concept), so it needs gentle steering. The `to_out[0]` representations are more concept-specific and tolerate stronger intervention.

---

## 4. Key Advantages of Our Approach

Compared to traditional unlearning methods (ESD, SalUn, TRACE, etc.), steering vectors offer:

1. **No model retraining required** -- vectors are learned via forward passes only (no gradient computation on model parameters).
2. **Inference-time application** -- the base model remains unchanged; steering is applied/removed at will.
3. **Modular and composable** -- different concept vectors can be applied independently or combined.
4. **Fast computation** -- learning vectors requires only ~100 forward passes (50 prompt pairs x 2), not iterative fine-tuning.
5. **Per-component control** -- `pincer_v2`'s per-component beta allows independent tuning of source-level vs output-level steering strength, enabling precise concept removal with minimal collateral damage.

---

## 5. Implementation Details

### 5.1 Diverse Prompt Pairs (CASteer Methodology)

Both modes use 50 ImageNet classes as diverse base contexts. For objects:
```
Positive: "tench with Dog", "goldfish with Dog", "tiger shark with Dog", ...
Negative: "tench", "goldfish", "tiger shark", ...
```
For styles:
```
Positive: "tench, Van Gogh style", "goldfish, Van Gogh style", ...
Negative: "tench", "goldfish", ...
```

Averaging across 50 diverse contexts cancels out context-specific features (layout, composition, colour palette), isolating only the target concept direction.

### 5.2 Mask-Aware Pooling

For layers that process T5 token sequences (used in `hybrid` mode), padding tokens are excluded via the T5 attention mask before averaging. This prevents the steering vector from being diluted by meaningless padding activations.

### 5.3 Vector Storage Format

Vectors are saved as PyTorch tensors organised by layer name and timestep:
```python
vectors = {
    "time_text_embed": {0: tensor(3072,), 1: tensor(3072,), ...},
    "double_7": {2: tensor(3072,), ...},
    ...
}
```

Filenames follow the convention: `{Concept}_{mode}_diverse_vectors.pt`

### 5.4 Notebook Cell Structure

| Cell | Purpose |
|---|---|
| 0 | Markdown introduction and metric definitions |
| 1 | Package installations (diffusers, transformers, CLIP, clean-fid) |
| 2 | Imports, configuration, prompt generation functions |
| 3 | `FluxSteering` class (2 modes: `hybrid` + `pincer_v2`) |
| 3b | `verify_text_entry_points()` -- empirical proof of text injection points |
| 4 | `QualityMetrics` class (FID, CLIP score) |
| 4B | `LLaVAClassifier` (VLM-based image classification) |
| 5 | `UnlearnCanvasEvaluator` (orchestrates UA/IRA/CRA evaluation) |
| 6 | Model loading + `FluxSteering` initialisation |
| 7 | Experiment configuration (target concept, beta, mode) |
| 8 | Learn steering vectors (diverse CASteer methodology) |
| 8B | Quick diagnostic test (zeroing tests + beta sweep grid) |
| 9 | Full UnlearnCanvas evaluation (UA, IRA, CRA) |
| 10 | Quality metrics (FID, CLIP score) |
| 11 | Results compilation and CSV export |
| 12 | Comparison table vs baselines (FLUX and SD1.5) |
| 13a | Full benchmark (loops over all 10 styles) |
| 13b | Visualisation (baseline vs steered side-by-side) |

---

## 6. Current Status and Next Steps

- The pipeline is fully implemented with two refined steering modes (`hybrid` for styles, `pincer_v2` for objects).
- Per-component beta in `pincer_v2` enables independent control of CLIP source steering vs attention output steering.
- Diagnostic experiments confirm the architectural insight: object identity flows through CLIP -> modulation, not T5 -> attention.
- Qualitative results show successful cat and dog removal with scene preservation at appropriate beta settings.
- Full UnlearnCanvas benchmark infrastructure is in place with LLaVA classification, FID/CLIP metrics, comparison tables, and resume support.
- Results are saved in CSV format (`benchmark_results.csv`) with per-concept UA, IRA, CRA metrics.
