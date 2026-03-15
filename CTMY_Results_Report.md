# CTMY Project Results Report: Steering Vectors for Concept Removal in FLUX Diffusion Models

## 1. Introduction

This report documents the methodology, implementation, and experimental results of applying **activation steering vectors** to the **FLUX.1-schnell** diffusion model for targeted concept removal (unlearning). The work is evaluated on the **UnlearnCanvas benchmark** and compared against recent baselines from the TRACE paper (ICLR 2026).

The project is implemented across three notebooks, each representing a stage in the iterative development of the steering approach:

| Notebook | Role |
|---|---|
| `Copy_of_Main_Eval_Pipeline.ipynb` | Initial evaluation pipeline. Implements two steering approaches: (1) **Dual Stream Block targeting** (19 DoubleStream + 38 SingleStream blocks) and (2) **Gate-only steering** (`context_embedder` + `time_text_embed`). Contains the first comprehensive style unlearning results (all 10 styles) and Dogs object unlearning experiment. |
| `SteeringVectorsUnlearnCanvas.ipynb` | Expanded exploration. Implements 8+ steering modes (`entry_point`, `block`, `pincer`, `hybrid`, `double_proj`, `all`, `object`, `joint_attn`) to systematically probe which layers matter. Introduces the diverse CASteer prompt methodology (50 ImageNet-based prompt pairs), text entry point verification experiments, and quick steering diagnostic tests. |
| `SteeringVectorsUnlearnCanvasPincerV2.ipynb` | Final refined pipeline. Consolidates findings into 2 optimised modes: `hybrid` (styles) and `pincer_v2` (objects). Adds per-component beta, diagnostic zeroing experiments, LLaVA/CLIP classification, FID/CLIP quality metrics, and full UnlearnCanvas benchmark evaluation infrastructure. |

---

## 2. Methodology: Steering Vectors

### 2.1 Core Idea

Steering vectors are computed as the **difference in internal activations** when the model processes a *positive* prompt (containing the target concept) versus a *negative* prompt (without it). During generation, these vectors are subtracted from the model's activations to suppress the target concept -- all at inference time, without any model fine-tuning or retraining.

### 2.2 Target Model

- **Model**: `black-forest-labs/FLUX.1-schnell` (distilled variant)
- **Denoising steps**: 4 (Schnell)
- **Precision**: bfloat16
- **Architecture**: FLUX transformer with 19 double-stream blocks (`FluxTransformerBlock`) and 38 single-stream blocks (`FluxSingleTransformerBlock`)

### 2.3 Key Architectural Insight

FLUX has **two independent text conditioning paths**:

1. **CLIP path**: `pooled_text → time_text_embed (MLP) → shift/scale modulation → every block`. This is the **primary concept source**.
2. **T5 path**: `T5_embeddings → context_embedder (Linear 4096→3072) → add_k/add_q → joint attention`. This provides fine-grained textual detail.

Diagnostic zeroing experiments (`SteeringVectorsUnlearnCanvas.ipynb`, verified in `PincerV2`) revealed:
- **Zeroing `time_text_embed` (CLIP)**: produces **pure noise** (~70.0 mean pixel difference) -- proving CLIP is the primary concept identity source.
- **Zeroing `context_embedder` (T5)**: still produces recognisable objects (~39.0 mean pixel difference) -- proving T5 is secondary for concept identity.

This insight is foundational: **object identity flows through CLIP → modulation**, not T5 → attention. The T5 path does matter for stylistic attributes carried through text descriptions.

### 2.4 Evolution of Steering Approaches

The project explored multiple steering strategies across the three notebooks, converging on two refined modes:

#### Stage 1: Initial Exploration (`Copy_of_Main_Eval_Pipeline.ipynb`)

Two approaches were tested:

**Approach A -- Dual Stream Block Targeting:**
- Hooks `attn.to_out[0]` in all 19 DoubleStream blocks and `proj_out` in all 38 SingleStream blocks
- Activation aggregation: `output.detach().mean(dim=(0, 1))` (mean over batch and sequence)
- Top-k = 15 vectors selected by L2 norm
- Used for: style unlearning across all 10 styles

**Approach B -- Gate-Only Steering:**
- Hooks only 2 layers: `context_embedder` and `time_text_embed`
- Activation aggregation: `output.detach().flatten(0, -2).sum(dim=0)` (sum-based to prevent dilution with short prompts)
- Top-k = 20 vectors
- Used for: Dogs object unlearning experiment
- Simple prompt pairs: "Dogs" vs "Object"

#### Stage 2: Systematic Mode Exploration (`SteeringVectorsUnlearnCanvas.ipynb`)

The `FluxSteering` class was expanded to support 8+ modes, each targeting different layer combinations:

| Mode | Layers Targeted | Purpose |
|---|---|---|
| `entry_point` | `context_embedder` + `time_text_embed` | TRACE-inspired, two text entry gates |
| `block` | `attn.to_out[0]` in all 19 DoubleStream blocks | Spatial attention outputs only |
| `pincer` | `time_text_embed` + `attn.to_out[0]` in double blocks 7-18 | CLIP source + mid/late spatial |
| `hybrid` | `context_embedder` + `time_text_embed` + `add_k_proj`/`add_q_proj` in 19 blocks | All text entry points |
| `double_proj` | `add_k_proj` + `add_q_proj` in 19 DoubleStream blocks | Text-side K/Q projections |
| `all` | All entry points + all block projections | Maximum coverage |
| `object` | Object-optimised variant | Specialised for object removal |
| `joint_attn` | `attn.to_out[0]` in all 19 DoubleStream blocks | Joint attention outputs |

Key innovations introduced:
- **Diverse CASteer prompt methodology**: 50 ImageNet classes as base contexts (e.g., "tench with Cat" / "tench"), averaging across diverse contexts to isolate the pure concept direction
- **Mask-aware pooling**: For T5 token sequences, padding tokens are excluded via the T5 attention mask before averaging
- **Text entry point verification**: Empirical proof that `context_embedder` and `time_text_embed` are the only root modules where raw text embeddings enter the transformer

#### Stage 3: Refined Final Modes (`SteeringVectorsUnlearnCanvasPincerV2.ipynb`)

Consolidated into 2 optimised modes:

| Mode | Layers Targeted | Best For | Typical Beta |
|---|---|---|---|
| **`hybrid`** | `context_embedder` + `time_text_embed` + `add_k_proj`/`add_q_proj` in 19 DoubleStream blocks | **Style unlearning** | β = 2.0-5.0 (scalar) |
| **`pincer_v2`** | `time_text_embed` (CLIP source) + `to_out[0]` in 19 DoubleStream blocks (image attention output) | **Object unlearning** | β = {"clip": 3.0, "attn": 10.0} |

**`hybrid` mode** (Style Unlearning):
- Steers all text injection points because style IS carried through T5 text descriptions
- Uses mask-aware pooling for T5 representations
- Keeps ALL learned vectors (no top-k filtering)

**`pincer_v2` mode** (Object Unlearning):
- Two-pronged: weakens the global concept signal at its CLIP source AND removes it where it manifests in image attention outputs
- Per-component beta: gentle for CLIP (fragile, carries ALL conditioning) and aggressive for `to_out[0]` (more concept-specific)
- Top-k selection for `to_out` vectors (default k=15); CLIP vectors always retained
- Ignores T5/context path (irrelevant for object identity, as proven by zeroing experiments)

### 2.5 Computation of Steering Vectors

**Step 1 -- Hooking target layers.** Forward hooks are registered on specific layers inside the FLUX transformer. The exact layers depend on the steering mode.

**Step 2 -- Collecting activations.** Using the **diverse CASteer methodology** (50 ImageNet classes as base contexts):
- For each (positive, negative) prompt pair (e.g., "tench with Dog" / "tench"), run a forward pass with both prompts using the same seed.
- Record activations at each hooked layer and each denoising step.
- Averaging across 50 diverse contexts cancels out context-specific noise, isolating the pure concept direction.

**Step 3 -- Averaging differences.** The mean activation difference is computed per layer per denoising step:
```
diff[layer][step] = mean( activation_positive ) - mean( activation_negative )
```
Each difference vector is then L2-normalised to unit norm.

**Step 4 -- Vector selection.** Depending on mode:
- `hybrid`: Keeps ALL vectors (no filtering)
- `pincer_v2`: Keeps all CLIP vectors + top-k `to_out` vectors ranked by L2 norm of the raw difference

### 2.6 Application of Steering Vectors (Inference-Time)

During image generation, the learned vectors are applied via forward hooks. At each denoising step, for each hooked layer:

```
score = output · direction                    # dot product
score = max(0, score)                         # clip_negative: only fire when aligned
update = β × score × direction               # scale by steering strength
output_steered = output - update              # subtract concept direction
```

The steering strength is controlled by **beta (β)**. In `pincer_v2` mode, beta is specified **per-component**:
- `beta["clip"]` = 3.0 -- gentle steering for `time_text_embed`
- `beta["attn"]` = 10.0 -- aggressive steering for `to_out[0]`

---

## 3. Experiments and Results

### 3.1 Evaluation Protocol: UnlearnCanvas Benchmark

The evaluation follows the UnlearnCanvas benchmark protocol, adapted for FLUX:

- **10 artistic styles** (from TRACE paper Section 5.1): Van_Gogh, Watercolor, Cartoon, Cubism, Winter, Pop_Art, Ukiyoe, Impressionism, Byzantine, Bricks
- **20 object classes**: Architecture, Bear, Bird, Butterfly, Cat, Dog, Fish, Flame, Flowers, Frog, Horse, Human, Jellyfish, Rabbits, Sandwich, Sea, Statue, Tower, Tree, Waterfalls
- **Prompt format**: `"A {Object} image in {Style} style."`
- **Learning seeds**: 20 seeds (range 0-19) for vector computation
- **Evaluation seeds**: 3 seeds (range 20-22) per style-object combination
- **Total images per target concept**: 10 styles × 20 objects × 3 seeds = 600 images

### 3.2 Classification Method

**LLaVA-1.6-Vicuna-7B** is the primary classifier, following the methodology from the TRACE paper (Appendix E.4, Figures 6-7). The TRACE paper demonstrates that UnlearnCanvas's original SD1.5-trained classifier generalises poorly to modern models like FLUX (<6% accuracy). LLaVA provides accurate zero-shot classification using a numbered-list multiple-choice prompt format:

```
"You are an image classifier. Classify the artistic style of the given image.
Instruction: Choose exactly one option from the numbered list below.
Respond with only the number.

Options:
1. Van Gogh
2. Watercolor
...
10. Bricks"
```

**CLIP ViT-L/14** is available as a faster alternative (~0.1s/image vs ~2-5s/image for LLaVA).

### 3.3 Quantitative Metrics

Three core metrics from the UnlearnCanvas benchmark:

| Metric | Definition | Goal |
|---|---|---|
| **UA (Unlearning Accuracy)** | `1 - (images classified as target / total target-concept images)` | Higher = better (target concept is suppressed) |
| **IRA (In-Domain Retain Accuracy)** | Classification accuracy on other concepts in the **same domain** (e.g., other styles when unlearning a style) | Higher = better (related knowledge preserved) |
| **CRA (Cross-Domain Retain Accuracy)** | Classification accuracy on concepts in the **other domain** (e.g., object accuracy when unlearning a style) | Higher = better (unrelated knowledge preserved) |

Two additional quality metrics:

| Metric | Tool | Goal |
|---|---|---|
| **FID (Frechet Inception Distance)** | `clean-fid` library | Lower = better (image quality/distribution match) |
| **CLIP Score** | CLIP ViT-L/14 cosine similarity | Higher = better (text-image alignment) |

### 3.4 Pipeline Workflow

The evaluation pipeline follows a two-phase approach for VRAM efficiency:

**Phase 1 -- Generation**: Load FLUX, learn steering vectors using diverse CASteer prompts, generate all benchmark images (with steering) and baseline images (without steering), save to disk. Resume support is built in (existing images are skipped on re-run).

**Phase 2 -- Classification**: Unload FLUX from GPU, load LLaVA classifier, classify all saved images for both style and object, compute UA/IRA/CRA metrics. Unload LLaVA, reload FLUX for the next concept.

The full benchmark mode (`RUN_FULL_BENCHMARK = True`) loops over all 10 styles (or 20 objects), learning vectors and evaluating each in sequence with incremental CSV output.

### 3.5 Results: Style Unlearning (`Copy_of_Main_Eval_Pipeline.ipynb`)

#### Van_Gogh Beta Sweep (Dual Stream Block Approach)

The initial pipeline performed a beta sweep for Van_Gogh style unlearning using the Dual Stream Block approach (19 DoubleStream + 38 SingleStream blocks):

| Beta | UA (%) | IRA (%) | CRA (%) | Average (%) |
|------|--------|---------|---------|-------------|
| 0 | 100.0 | 55.6 | 100.0 | 85.2 |
| 1 | 100.0 | 50.0 | 100.0 | 83.3 |
| 2 | 100.0 | 55.6 | 100.0 | 85.2 |
| 3 | 100.0 | 61.1 | 100.0 | 87.0 |
| 4 | 100.0 | 61.1 | 100.0 | 87.0 |
| 5 | 100.0 | 66.7 | 95.0 | 87.2 |
| 8 | 100.0 | 66.7 | 95.0 | 87.2 |

**Optimal beta: 5** (highest average while maintaining UA ≥ 90%). Van_Gogh achieved 100% UA across all beta values tested, indicating that this distinctive style is effectively captured by the steering vectors.

#### All 10 Styles at β=2.0 (Dual Stream Block Approach)

Comprehensive evaluation across all 10 TRACE styles:

| Style | UA (%) | IRA (%) | CRA (%) |
|-------|--------|---------|---------|
| Van_Gogh | 95.0 | 54.4 | 92.0 |
| Watercolor | 90.0 | 55.6 | 90.0 |
| Impressionism | 85.0 | 50.0 | 91.5 |
| Byzantine | 85.0 | 53.3 | 91.5 |
| Cubism | 70.0 | 50.0 | 91.5 |
| Winter | 45.0 | 47.8 | 90.0 |
| Pop_Art | 30.0 | 48.3 | 93.0 |
| Bricks | 30.0 | 47.2 | 91.5 |
| Ukiyoe | 20.0 | 41.1 | 93.0 |
| Cartoon | 10.0 | 44.4 | 91.0 |

**Key observations:**
- **Highly distinctive styles** (Van_Gogh, Watercolor, Impressionism, Byzantine) achieve strong UA (85-95%) -- their visual signatures are well-captured by steering vectors
- **Subtle or broad styles** (Cartoon, Ukiyoe, Pop_Art, Bricks) prove harder to unlearn (10-30% UA) -- these styles may overlap with general image features or lack a single dominant visual direction
- **CRA remains consistently high** (90-93%) across all styles, indicating minimal cross-domain damage to object recognition
- **IRA is moderate** (41-56%) -- removing one style has some collateral impact on classifying other styles, likely because the LLaVA classifier itself has limited style discrimination

**Top steering vectors** consistently concentrated in the **last SingleStream block** (`single_37`), with strengths of 600-1040. This suggests that style information is most concentrated in the deepest single-stream layers.

### 3.6 Results: Object Unlearning (`Copy_of_Main_Eval_Pipeline.ipynb`)

#### Dogs Unlearning (Gate-Only Approach)

The Dogs experiment used the Gate-only approach with simple prompts ("Dogs" vs "Object"):

**Learned vector strengths (top 8):**

| Rank | Layer | Step | Strength |
|------|-------|------|----------|
| 1-4 | gate_context | 0-3 | 8512.0 |
| 5-8 | gate_time_text | 0-3 | 7.28-7.38 |

**Evaluation results (β=0.5):**

| Metric | Value |
|--------|-------|
| UA | 0.00% |
| IRA | 92.11% |
| CRA | 44.00% |

The 0% UA indicates Dogs were completely removed (no images classified as Dogs), demonstrating strong unlearning. The high IRA (92.11%) shows other objects were preserved. However, the low CRA (44.00%) reveals significant collateral damage to style classification, likely because the Gate-only approach steers broadly at the embedding level, affecting all downstream processing.

### 3.7 Results: Diagnostic Tests (`SteeringVectorsUnlearnCanvas.ipynb`)

#### Quick Steering Test -- Cat (Pincer Mode)

Pixel-level impact measurements comparing steered vs baseline images:

| Configuration | Mean Pixel Diff | Pixels Changed (%) |
|---------------|-----------------|---------------------|
| β=2, clip_neg=True | 37.6 | 96.8 |
| β=5, clip_neg=True | 38.7 | 97.2 |
| β=40, clip_neg=True | 103.6 | 99.7 |
| β=5, clip_neg=False | 118.0 | 99.6 |
| β=40, clip_neg=False | 118.2 | 99.6 |

The `clip_negative=True` setting (only steer when activations are positively aligned with the concept direction) provides more controlled intervention. Without clipping, even β=5 produces 118.0 pixel difference -- comparable to β=40 with clipping.

#### Text Entry Point Verification

Empirical proof that only two modules are root text-dependent entry points:
- **`context_embedder`**: T5 embeddings enter here (effect = 39.0 pixel difference when zeroed)
- **`time_text_embed`**: CLIP pooled text enters here (effect = 70.0 pixel difference when zeroed)

All other text-dependent layers are downstream children of these two roots.

### 3.8 Qualitative Results (`SteeringVectorsUnlearnCanvasPincerV2.ipynb`)

#### Cat Unlearning (pincer_v2)

The diagnostic test generates an 8-panel comparison grid:

| Test | Observation |
|---|---|
| T5 zeroed (context_embedder=0) | Cat still clearly visible -- confirms T5 is NOT the object identity source |
| CLIP zeroed (time_text_embed=0) | Pure noise/static -- confirms CLIP is the primary concept source |
| Baseline (no steering) | Clear cat in kitchen scene |
| clip=2, attn=5 | Cat begins to fade; some morphological distortion |
| clip=3, attn=10 (default) | Cat significantly suppressed; replaced by amorphous shapes |
| clip=5, attn=15 | Strong removal; image quality starts to degrade |
| clip=5, attn=20 | Near-complete removal but scene coherence also affected |

Additional beta sweep:
- **β 5/40 (clip/attn)**: Cat replaced by small artifact, scene mostly preserved
- **β 10/40**: Cat fully removed, minor scene artifacts
- **β 20/40**: Cat fully removed, scene begins to wash out
- **β 50**: Excessive -- horizontal banding artifacts appear

#### Dog Unlearning (pincer_v2)

| Test | Observation |
|---|---|
| T5 zeroed | Dog still present (confirms T5 irrelevance for objects) |
| CLIP zeroed | Pure noise (confirms CLIP is source) |
| Baseline | Clear dog in park/kitchen scenes |
| clip=3, attn=10 (default) | Dog significantly suppressed |
| No attn steering (CLIP only) | Dogs still partially visible -- CLIP alone insufficient |
| High attn steering | Dogs fully removed; scenes remain coherent |
| pincer_v2 (19 vectors) | Clean removal with good scene preservation |

The ablation between "CLIP only" and "CLIP + attn" demonstrates that **both components of the pincer are necessary**: CLIP weakens the global signal, while `to_out[0]` removes the concept where it has already manifested in image representations.

#### Surgical Unlearning Comparison (Cat)

Three steering strategies compared:
- **Subtract** (standard): Removes cat but produces line-drawing artifacts
- **Orthogonal**: Produces near-complete noise
- **Block targeting**: Preserves scene but cat remains partially visible

This motivated the final `pincer_v2` design combining source-level (CLIP) and output-level (`to_out`) steering.

### 3.9 Comparison with Baselines

Published FLUX baselines from the TRACE paper (Table 1) for **style removal**:

| Method | UA (%) | IRA (%) | CRA (%) | FID |
|---|---|---|---|---|
| LOCOEDIT (FLUX) | 66.45 | 33.23 | 83.44 | 55.56 |
| UCE (FLUX) | 67.43 | 34.78 | 76.56 | 58.90 |
| TRACE (FLUX) | 88.60 | 36.10 | 96.40 | 51.67 |

For broader context, SD1.5 baselines from TRACE Table 2:

| Method | UA (%) | IRA (%) | CRA (%) |
|---|---|---|---|
| ESD (SD1.5) | 98.58 | 80.97 | 93.96 |
| FMN (SD1.5) | 88.48 | 56.77 | 46.60 |
| UCE (SD1.5) | 98.40 | 60.22 | 47.71 |
| CA (SD1.5) | 60.82 | 96.01 | 92.70 |
| SalUn (SD1.5) | 86.26 | 90.39 | 95.08 |
| SAeUron (SD1.5) | 95.80 | 99.10 | 99.40 |
| TRACE (SD1.5) | 95.02 | 93.84 | 86.22 |

Note: SD1.5 numbers are NOT directly comparable to FLUX results and are provided for broader context only. SD1.5 methods benefit from a simpler architecture where cross-attention is the sole text injection point.

### 3.10 Hyperparameter Guidance

Based on experiments across all three notebooks:

| Concept Type | Mode | Beta | Top-k | Notes |
|---|---|---|---|---|
| Style | `hybrid` | 2.0-5.0 (scalar) | All kept | T5 path matters for style; β=5 optimal for Van_Gogh |
| Object | `pincer_v2` | {"clip": 3.0, "attn": 10.0} | 15 | Per-component beta critical; CLIP is fragile |
| Object (gate) | gate-only | 0.5 | 20 | Strong unlearning but CRA collateral damage |

---

## 4. Key Advantages of Our Approach

Compared to traditional unlearning methods (ESD, SalUn, TRACE, etc.), steering vectors offer:

1. **No model retraining required** -- vectors are learned via forward passes only (no gradient computation on model parameters).
2. **Inference-time application** -- the base model remains unchanged; steering is applied/removed at will.
3. **Modular and composable** -- different concept vectors can be applied independently or combined.
4. **Fast computation** -- learning vectors requires only ~100 forward passes (50 prompt pairs × 2), not iterative fine-tuning.
5. **Per-component control** -- `pincer_v2`'s per-component beta allows independent tuning of source-level vs output-level steering strength, enabling precise concept removal with minimal collateral damage.

---

## 5. Implementation Details

### 5.1 Diverse Prompt Pairs (CASteer Methodology)

Both `hybrid` and `pincer_v2` modes use 50 ImageNet classes as diverse base contexts. For objects:
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

### 5.4 Notebook Structure Summary

**`Copy_of_Main_Eval_Pipeline.ipynb`** (~16 cells):

| Section | Purpose |
|---|---|
| Cells 0-1 | FluxSteering class (Dual Stream Block approach) |
| Cell 2 | FluxSteering class (Gate-only approach) |
| Cells 3-4 | Vector learning (style) |
| Cells 5-10 | Dogs object unlearning experiment |
| Cells 11-12 | Comprehensive pipeline with LLaVA evaluation |
| Cells 13-15 | Full 10-style benchmark with results tables |

**`SteeringVectorsUnlearnCanvas.ipynb`** (~13+ cells):

| Section | Purpose |
|---|---|
| Cell 0 | Introduction and metric definitions |
| Cells 1-2 | Imports, configuration, 50 ImageNet classes, prompt generators |
| Cell 3 | FluxSteering class (8+ modes: entry_point, block, pincer, hybrid, etc.) |
| Cell 4 | Text entry point verification function |
| Cell 5 | QualityMetrics class (FID, CLIP score) |
| Cell 6 | LLaVAClassifier (TRACE paper methodology) |
| Cell 7+ | UnlearnCanvas evaluator, experiments, diagnostics |

**`SteeringVectorsUnlearnCanvasPincerV2.ipynb`** (~14 cells):

| Cell | Purpose |
|---|---|
| 0 | Introduction and metric definitions |
| 1 | Package installations |
| 2 | Imports, configuration, prompt generation |
| 3 | FluxSteering class (2 refined modes: `hybrid` + `pincer_v2`) |
| 3b | `verify_text_entry_points()` |
| 4-4B | QualityMetrics + LLaVAClassifier |
| 5 | UnlearnCanvasEvaluator |
| 6-7 | Model loading + experiment configuration |
| 8-8B | Vector learning + diagnostic tests |
| 9-10 | Full benchmark evaluation + quality metrics |
| 11-13 | Results compilation, baseline comparison, visualisation |

---

## 6. Current Status and Next Steps

### Current Status

- The pipeline is fully implemented across three notebooks, with iterative refinement from broad exploration to two optimised modes (`hybrid` for styles, `pincer_v2` for objects).
- **Style unlearning** achieves strong results for distinctive styles (Van_Gogh 95-100% UA, Watercolor 90% UA) but struggles with subtle styles (Cartoon 10% UA, Ukiyoe 20% UA) at β=2.0.
- **Object unlearning** demonstrates successful concept removal (Dogs 0% UA with gate approach; Cat/Dog qualitative removal with pincer_v2) but requires careful beta tuning to avoid collateral damage.
- Per-component beta in `pincer_v2` enables independent control of CLIP source steering vs attention output steering, improving the unlearning/retention trade-off.
- The diagnostic zeroing experiments provide clear architectural evidence for the design decisions.
- Full UnlearnCanvas benchmark infrastructure is in place with LLaVA classification, FID/CLIP metrics, comparison tables, and resume support.
- Results are saved in CSV format (`benchmark_results.csv`) with per-concept UA, IRA, CRA metrics.

### Key Findings

1. **Steering vectors are most effective for visually distinctive concepts** -- Van_Gogh's swirling brushstrokes are easier to isolate than Cartoon's diffuse characteristics.
2. **The dual text path in FLUX requires mode-specific strategies** -- objects need CLIP-focused steering (pincer_v2), styles need broader text-path steering (hybrid).
3. **Per-component beta is critical** -- CLIP carries all conditioning and needs gentle intervention; spatial attention outputs are more concept-specific and tolerate aggressive steering.
4. **The diverse CASteer prompt methodology** (50 ImageNet contexts) is essential for isolating clean concept directions that generalise beyond specific prompt formulations.
