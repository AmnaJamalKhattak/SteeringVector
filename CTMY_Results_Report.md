# CTMY Project Results Report: Steering Vectors for Concept Removal in Diffusion Models

## 1. Introduction

This report summarises the current progress on applying **activation steering vectors** to the **FLUX.1** diffusion model for targeted concept removal (unlearning). The work is evaluated on the **UnlearnCanvas benchmark** and compared against recent baselines from the TRACE paper (ICLR 2026). Two notebooks constitute the primary codebase:

| Notebook | Role |
|---|---|
| `Copy_of_Main_Eval_Pipeline.ipynb` | Initial exploration pipeline with multiple `FluxSteering` class variants, LLaVA-based evaluation, beta-sweep, and full benchmark orchestration. |
| `SteeringVectorsUnlearnCanvas.ipynb` | Comprehensive, production-ready pipeline with 9 steering modes, FID/CLIP quality metrics, comparison tables, and full UnlearnCanvas protocol evaluation. |

---

## 2. Methodology: Steering Vectors

### 2.1 Core Idea

Steering vectors are computed as the **difference in internal activations** when the model processes a *positive* prompt (containing the target concept) versus a *negative* prompt (without the target concept). During generation, these vectors are subtracted from the model's activations to suppress the target concept, without any model fine-tuning or retraining.

### 2.2 Computation of Steering Vectors

**Step 1 -- Hooking target layers.** Forward hooks are registered on specific layers inside the FLUX transformer. The exact layers depend on the steering mode (see Section 2.4).

**Step 2 -- Collecting activations.** For each (positive, negative) prompt pair and across multiple seeds:
- Run a forward pass with the positive prompt; record activations at each hooked layer and each denoising step.
- Run a forward pass with the negative prompt; record activations at each hooked layer and each denoising step.

**Step 3 -- Averaging differences.** The mean activation difference is computed per layer per denoising step:

```
diff[layer][step] = mean( activation_positive ) - mean( activation_negative )
```

Each difference vector is then L2-normalised to unit norm.

**Step 4 -- Top-k selection.** Vectors are ranked by gradient norm (magnitude of the raw difference before normalisation). Only the top-k most impactful (layer, step) pairs are retained.

**Diverse learning (CASteer methodology).** The `SteeringVectorsUnlearnCanvas` notebook implements `learn_vectors_diverse()`, which uses 50 diverse prompt pairs (e.g., "a tench with a Dog" / "a tench" across many context nouns). This cancels out context and isolates the pure concept direction. The `Copy_of_Main_Eval_Pipeline` notebook uses a simpler single-pair approach (e.g., "Dogs" / "Object") across 20 seeds.

### 2.3 Application of Steering Vectors (Inference-Time)

During image generation, the learned vectors are applied via forward hooks. At each denoising step, for each hooked layer:

1. Compute the **projection score**: `score = output @ target_direction` (dot product along the steering direction).
2. Optionally **clip** negative scores to zero (`clip_negative` mode) -- ensures steering only fires when the activation is aligned with the concept direction.
3. Compute the **update**: `update = (beta * score).unsqueeze(-1) * target_direction`.
4. **Subtract** the update from the layer output: `output = output - update`.

Some modes additionally apply **norm preservation** (renormalise the steered output to match the original output norm) to prevent magnitude drift.

The steering strength is controlled by the hyperparameter **beta (β)**. Higher beta values produce stronger concept removal but risk degrading unrelated content.

### 2.4 Steering Modes

The `SteeringVectorsUnlearnCanvas` notebook implements 9 distinct steering modes, targeting different subsets of the FLUX transformer:

| Mode | Layers Targeted | Best For |
|---|---|---|
| `entry_point` | `context_embedder` (T5 → model), `time_text_embed` (CLIP + timestep MLP) | General |
| `block` | `attn.to_out[0]` (double-stream), `proj_out` (single-stream) | General (CASteer-style) |
| `double_proj` | `add_k_proj`, `add_q_proj` in 19 double-stream blocks | Object unlearning (EraseAnything-style) |
| `all` | Union of `entry_point` + `double_proj` + double-stream `to_out` | Maximum coverage |
| **`hybrid`** | Entry points + `add_k_proj`/`add_q_proj` | **Style unlearning (best)** |
| `object` | Image-side representations (`add_v`, `to_out`, `proj_out` image slice) | Object unlearning |
| `object_v2` | Entry points + attention output + single-stream image slice | Object unlearning (alt) |
| `joint_attn` | Double-stream `attn.to_out[0]` only | Object unlearning |
| **`pincer`** | `time_text_embed` (CLIP) + double-stream `attn.to_out[0]` (layers 7-18) | **Object unlearning (best)** |

The **`hybrid`** mode is identified as best for style unlearning (typical β = 2.0), while **`pincer`** mode is identified as best for object unlearning (typical β = 10.0-15.0). The `pincer` mode deliberately excludes `context_embedder` (T5) to prevent "ghosting" artefacts.

The `Copy_of_Main_Eval_Pipeline` notebook contains two `FluxSteering` class variants:
1. **Block-based**: hooks `attn.to_out[0]` in double-stream blocks and `proj_out` in single-stream blocks (CASteer-style).
2. **Gate-based**: hooks only the two global gates -- `context_embedder` and `time_text_embed` (inspired by the TRACE/CASteer papers).

### 2.5 Target Model

- **Model**: `black-forest-labs/FLUX.1-schnell` (distilled variant)
- **Denoising steps**: 4 (Schnell) or 28 (full model)
- **Precision**: bfloat16
- **Architecture**: FLUX transformer with 19 double-stream blocks (`FluxTransformerBlock`) and 38 single-stream blocks (`FluxSingleTransformerBlock`)

---

## 3. Experiments and Results

### 3.1 Evaluation Protocol: UnlearnCanvas Benchmark

The evaluation follows the UnlearnCanvas benchmark protocol, adapted for FLUX:

- **10 artistic styles**: Cartoon, Cubism, Winter, Pop Art, Ukiyoe, Impressionism, Byzantine, Van Gogh, Bricks, Watercolor
- **20 object classes**: Architectures, Bears, Birds, Butterfly, Cats, Dogs, Fishes, Flame, Flowers, Frogs, Horses, Human, Jellyfish, Rabbits, Sandwiches, Sea, Statues, Towers, Trees, Waterfalls
- **Prompt format**: `"A {Object} image in {Style} style."`
- **Evaluation seeds**: [42, 188, 999] (3 seeds per style-object combination)
- **Total images per target concept**: 10 styles x 20 objects x 3 seeds = 600 images

### 3.2 Classification Method

**LLaVA-1.6-Vicuna-7B** is used as the primary classifier, following the methodology from the TRACE paper (Appendix E.4). The TRACE paper demonstrates that UnlearnCanvas's original SD1.5-trained classifier generalises poorly to modern models like FLUX (<6% accuracy). LLaVA provides accurate zero-shot classification using a numbered-list multiple-choice prompt format.

The `SteeringVectorsUnlearnCanvas` notebook also supports CLIP ViT-L/14 as a faster alternative classifier (~0.1s/image vs ~2-5s/image for LLaVA), though LLaVA is the default due to higher accuracy.

### 3.3 Quantitative Metrics

Three core metrics from the UnlearnCanvas benchmark:

| Metric | Definition | Goal |
|---|---|---|
| **UA (Unlearning Accuracy)** | `1 - (images classified as target / total target-concept images)` | Higher is better (target concept is suppressed) |
| **IRA (In-Domain Retain Accuracy)** | Classification accuracy on other concepts in the **same domain** (e.g., other styles when unlearning a style) | Higher is better (related knowledge preserved) |
| **CRA (Cross-Domain Retain Accuracy)** | Classification accuracy on concepts in the **other domain** (e.g., object accuracy when unlearning a style) | Higher is better (unrelated knowledge preserved) |

Two additional quality metrics (in `SteeringVectorsUnlearnCanvas` only):

| Metric | Tool | Goal |
|---|---|---|
| **FID (Frechet Inception Distance)** | `clean-fid` library | Lower is better (image quality/distribution match) |
| **CLIP Score** | CLIP ViT-L/14 cosine similarity | Higher is better (text-image alignment) |

### 3.4 Pipeline Workflow

Both notebooks follow a two-phase approach for memory efficiency:

**Phase 1 -- Generation**: Load FLUX, learn steering vectors, generate all benchmark images (with steering) and baseline images (without steering), save to disk.

**Phase 2 -- Classification**: Unload FLUX from GPU, load LLaVA, classify all saved images, compute UA/IRA/CRA metrics. Resume support is built in (existing images are skipped on re-run).

The `Copy_of_Main_Eval_Pipeline` notebook orchestrates this via three stages:
- `run_learning_stage(STYLES)` -- learns and saves steering vectors for each style
- `run_generation_stage(STYLES, beta=2.0)` -- generates the full 10x20 image grid per target
- `run_evaluation_stage(STYLES, baseline_dir)` -- classifies and computes metrics

The `SteeringVectorsUnlearnCanvas` notebook runs per-concept (configurable via `TARGET_CONCEPT` and `TARGET_TYPE`) and includes a full-benchmark mode (`RUN_FULL_BENCHMARK = True`) that loops over all 10 styles.

### 3.5 Beta Sweep (Hyperparameter Tuning)

The `Copy_of_Main_Eval_Pipeline` includes an automatic beta sweep function (`run_full_style_beta_sweep`) that tests β values [0, 1, 2, 3, 4, 5, 8] and measures UA/IRA/CRA for each. The optimal β is selected by finding the value where UA ≥ 90% and then maximising IRA. Typical optimal values found:

- **Style unlearning**: β = 2.0
- **Object unlearning (pincer mode)**: β = 10.0-15.0
- **Object unlearning (joint_attn mode)**: β = 30.0

### 3.6 Comparison with Baselines

The `SteeringVectorsUnlearnCanvas` notebook includes comparison tables against published FLUX baselines from the TRACE paper (Table 1) for **style removal**:

| Method | UA (%) | IRA (%) | CRA (%) | FID |
|---|---|---|---|---|
| LOCOEDIT (FLUX) | 66.45 | 33.23 | 83.44 | 55.56 |
| UCE (FLUX) | 67.43 | 34.78 | 76.56 | 58.90 |
| TRACE (FLUX) | 88.60 | 36.10 | 96.40 | 51.67 |
| **Ours (Steering)** | **Evaluated per run** | **Evaluated per run** | **Evaluated per run** | **Evaluated per run** |

For broader context, SD1.5 baselines from TRACE Table 2 are also included (e.g., ESD: 98.58/80.97/93.96, SalUn: 86.26/90.39/95.08, SAeUron: 95.80/99.10/99.40), though these are not directly comparable to FLUX results.

### 3.7 Qualitative Evaluation

Both notebooks include **side-by-side visual comparison** of baseline (no steering) vs steered images. For style unlearning, comparisons show the target style applied to sample objects (Dogs, Cats, Birds). For object unlearning, comparisons show the target object rendered in sample styles (Van Gogh, Cartoon, Pop Art). These visualisations use the same seed and prompt for direct comparison.

The `SteeringVectorsUnlearnCanvas` notebook additionally includes a **quick steering diagnostic test** (Cell 8B) that:
- Runs destructive hook tests (zeroing out `context_embedder` and `time_text_embed` outputs) to empirically verify text-embedding entry points
- Generates images at multiple β values and `clip_negative` settings to quickly assess steering effectiveness before running the full evaluation

---

## 4. Key Advantages of Our Approach

Compared to traditional unlearning methods (ESD, SalUn, TRACE, etc.), steering vectors offer:

1. **No model retraining required** -- vectors are learned via forward passes only (no gradient computation on model parameters).
2. **Inference-time application** -- the base model remains unchanged; steering is applied/removed at will.
3. **Modular and composable** -- different concept vectors can be applied independently or combined.
4. **Fast computation** -- learning vectors requires only forward passes across a set of prompt pairs, not iterative fine-tuning.

---

## 5. Current Status and Next Steps

- The pipeline is fully implemented and operational across both notebooks.
- 9 steering modes have been explored, with `hybrid` and `pincer` identified as the best for style and object unlearning respectively.
- Full UnlearnCanvas benchmark evaluation infrastructure is in place, with LLaVA classification, FID, CLIP scores, beta sweeps, and comparison tables.
- Results are saved in CSV format (`benchmark_results.csv`) with columns: target concept, target type, beta, UA, IRA, CRA, CLIP score, FID, number of images, and timestamp.
- Resume support is built in throughout (both image generation and evaluation can be interrupted and resumed).
