# -*- coding: utf-8 -*-
"""CombinedUnlearnSD35.ipynb

Concept unlearning for Stable Diffusion 3.5 (MMDiT) via inference-time
activation steering. Mirrors combinedunlearnflux.py end-to-end so the
methodology section is unified across the two models. Architecture-
specific differences (3 text encoders vs 2, CFG vs no-CFG, pooled
embedding split) are absorbed inside SD35Steering.

Modes (parallel to FLUX):
  "pincer_v2"      -- STYLE  unlearning (β_pooled=0, β_t5 moderate)
  "pincer_perstep" -- OBJECT unlearning (per-step T5, β_pooled high)

Key adaptations vs FLUX:
  * Pipeline: StableDiffusion3Pipeline
  * N_STEPS = 28 (SD3.5 is not distilled like FLUX-schnell)
  * GUIDANCE_SCALE = 4.0 (lower than default 7.0 to let steering compete
    with the prompt-following force amplified by CFG; CFG=7 multiplies
    the conditional pathway 7x, partly undoing upstream steering)
  * Pooled embedding is 2048-d (CLIP-L pooled 768 + CLIP-G pooled 1280
    concatenated, NOT projected at this stage; projection happens inside
    time_text_embed.text_embedder)
  * context_embedder is a plain nn.Linear(joint_attention_dim=4096,
    caption_projection_dim). Output is (B, 333, caption_projection_dim)
    where 333 = 77 (CLIP-L+G) + 256 (T5) tokens. caption_projection_dim
    depends on the checkpoint:
       SD3 / SD3.5-medium : 1536 (24 heads x 64 dim)
       SD3.5-large        : 2432 (38 heads x 64 dim)
    The code reads this dynamically from activations -- no hardcoding.
  * Steering applied ONLY to the conditional batch position (out[1:]),
    not to the unconditional reference frame -- otherwise CFG arithmetic
    is corrupted and the steering effect partially cancels.

# Steering Vectors for SD3.5 - UnlearnCanvas Benchmark Evaluation
"""

# ============================================================================
# CELL 1: INSTALLATIONS
# ============================================================================

!pip install torch torchvision torchaudio --quiet
!pip install diffusers transformers accelerate sentencepiece --quiet
!pip install clean-fid --quiet
!pip install git+https://github.com/openai/CLIP.git --quiet
!pip install timm pandas matplotlib pillow tqdm --quiet

print("✓ All packages installed successfully!")

# ============================================================================
# CELL 2: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from collections import defaultdict
import matplotlib.pyplot as plt
from contextlib import contextmanager
from tqdm.auto import tqdm
import pandas as pd
import gc
from cleanfid import fid
import clip
from torchvision import transforms
import json
from datetime import datetime

# ============================================================================
# GOOGLE DRIVE SETUP (Optional - for Colab)
# ============================================================================
USE_GOOGLE_DRIVE = True
DRIVE_PATH = "/content/drive/MyDrive/UnlearnCanvas_SD35"

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(DRIVE_PATH, exist_ok=True)
        ROOT_DIR = DRIVE_PATH
        print(f"✓ Google Drive mounted at: {ROOT_DIR}")
    except Exception:
        print("⚠ Not in Colab or Drive mounting failed. Using local storage.")
        ROOT_DIR = "."
else:
    ROOT_DIR = "."

# ============================================================================
# HF auth (SD3.5 is gated -- needs an HF token)
# ============================================================================
try:
    from google.colab import userdata
    from huggingface_hub import login
    _hf_token = userdata.get("HF_TOKEN")
    if _hf_token:
        login(token=_hf_token)
        print("✓ HF authenticated via Colab secret HF_TOKEN")
except Exception:
    # Outside Colab: rely on `huggingface-cli login` having been run.
    pass

# ============================================================================
# UNLEARNCANVAS BENCHMARK CONFIGURATION (TRACE-aligned)
# ============================================================================
# 10 styles from TRACE paper for SD3.5 evaluation.
STYLES = [
    "Van_Gogh", "Watercolor", "Cartoon", "Cubism", "Winter",
    "Pop_Art", "Ukiyoe", "Impressionism", "Byzantine", "Bricks"
]

# 20 object classes (plural form to match TRACE's classifier label set).
OBJECTS = [
    "Architectures", "Bears", "Birds", "Butterfly", "Cats",
    "Dogs", "Fishes", "Flame", "Flowers", "Frogs",
    "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches",
    "Sea", "Statues", "Towers", "Trees", "Waterfalls"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {DEVICE}")

# Model configuration
# SD3.5-medium: faster, 24 transformer blocks
# SD3.5-large : higher quality, 38 transformer blocks. TRACE Table 1 values
#               likely come from the larger model.
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
N_STEPS = 28
GUIDANCE_SCALE = 4.0   # IMPORTANT: lower than default 7.0 so steering competes with CFG amplification
DTYPE = torch.bfloat16

# Steering vector configuration
LEARNING_SEEDS = list(range(0, 5))   # 5 seeds for hybrid-mode learning (unused by pincer modes)
EVAL_SEEDS = [188, 288, 588, 688, 888]   # TRACE's exact reproducibility seeds
GLOBAL_BETA = 5.0                    # placeholder; per-mode defaults override
TOP_K_VECTORS = 15

# ============================================================================
# IMAGENET CLASSES FOR DIVERSE PROMPT PAIRS (CASteer Appendix C)
# ============================================================================
IMAGENET_CLASSES = [
    "tench", "goldfish", "tiger shark", "hammerhead", "electric ray",
    "hen", "ostrich", "brambling", "goldfinch", "house finch",
    "junco", "indigo bunting", "robin", "bulbul", "jay",
    "magpie", "chickadee", "water ouzel", "kite", "bald eagle",
    "vulture", "great grey owl", "mud turtle", "box turtle", "banded gecko",
    "common iguana", "whiptail lizard", "agama", "frilled lizard", "alligator lizard",
    "green mamba", "thunder snake", "ringneck snake", "king snake", "garter snake",
    "vine snake", "trilobite", "scorpion", "black widow", "tarantula",
    "centipede", "grouse", "peacock", "quail", "partridge",
    "macaw", "lorikeet", "coucal", "bee eater", "hornbill"
]

def make_object_prompts(concept, num_prompts=50):
    """CASteer-style object prompt pairs.
    Pattern: ("class with concept", "class") -- bare class name negative.
    """
    n = min(num_prompts, len(IMAGENET_CLASSES))
    return [(f"{cls} with {concept}", f"{cls}") for cls in IMAGENET_CLASSES[:n]]

def make_style_prompts(concept, num_prompts=50):
    """CASteer-style style prompt pairs.
    Pattern: ("class, concept style", "class") -- bare class name negative.
    """
    n = min(num_prompts, len(IMAGENET_CLASSES))
    return [(f"{cls}, {concept} style", f"{cls}") for cls in IMAGENET_CLASSES[:n]]

NUM_DIVERSE_PROMPTS = 50    # CASteer saturation point (Appendix C.1 ablation)
RUN_FULL_BENCHMARK = False

# Directory structure (mirrors FLUX layout under a separate root)
for subdir in ["steering_vectors", "results", "baseline_images", "steered_images", "tables"]:
    os.makedirs(os.path.join(ROOT_DIR, subdir), exist_ok=True)

VECTOR_DIR   = os.path.join(ROOT_DIR, "steering_vectors")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
BASELINE_DIR = os.path.join(ROOT_DIR, "baseline_images")
STEERED_DIR  = os.path.join(ROOT_DIR, "steered_images")
TABLES_DIR   = os.path.join(ROOT_DIR, "tables")
RESULTS_CSV  = os.path.join(ROOT_DIR, "benchmark_results.csv")

print("\n" + "=" * 70)
print("SD3.5 UNLEARNCANVAS BENCHMARK CONFIGURATION")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Inference steps: {N_STEPS}")
print(f"Guidance scale:  {GUIDANCE_SCALE}  (lower than default 7.0 for steering to compete)")
print(f"Eval seeds:      {EVAL_SEEDS}")
print(f"Styles:          {len(STYLES)}")
print(f"Objects:         {len(OBJECTS)}")
print("=" * 70)

# ============================================================================
# CELL 3: SD35STEERING CLASS
# ============================================================================
"""
SD35Steering: inference-time concept removal in Stable Diffusion 3.5 (MMDiT).

ARCHITECTURE NOTES (vs FLUX):
  Three text encoders feed into the transformer:
    * CLIP-L (77 tok x 768d sequence + 768d pooled)
    * CLIP-G (77 tok x 1280d sequence + 1280d pooled)
    * T5     (256 tok x 4096d sequence)
  Pooled vector concatenates [CLIP-L pooled (768)] + [CLIP-G pooled (1280)]
  -> 2048d, fed to time_text_embed.
  Sequence context: CLIP-L+G concat (77 tok x 4096d) + T5 (256 tok x 4096d)
  concatenated along token axis (CLIP-then-T5 order, see pipeline source),
  fed to context_embedder = nn.Linear(4096 -> caption_projection_dim).
  Output: (B, 333, caption_projection_dim) where caption_projection_dim is
  1536 for medium / 2432 for large. The code below reads dimensions
  dynamically from hook activations.

  CFG: SD3.5 is not distilled; runs at guidance_scale ~ 4-7 with batched
  unconditional + conditional. context_embedder fires once per step over
  a batch of size 2 (uncond=0, cond=1). Steering must apply only to the
  conditional position to preserve the CFG reference frame; touching
  uncond contaminates the reference and can amplify the concept rather
  than remove it.

Modes (parallel to FLUX):
  "pincer_v2"      -- STYLE  : single direction over time, β_pooled small
  "pincer_perstep" -- OBJECT : per-step directions, β_pooled high
"""

class SD35Steering:
    VALID_MODES = ("pincer_v2", "pincer_perstep")

    def __init__(self, pipe, device="cuda", n_steps=28, mode="pincer_perstep",
                 guidance_scale=4.0):
        self.pipe = pipe
        self.device = device
        self.n_steps = n_steps
        self.mode = mode
        self.guidance_scale = guidance_scale
        self._current_step = -1
        self._handles = []
        self._current_attention_mask = None

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.VALID_MODES}.")

        # Layer references
        self.target_layers = {
            "context_embedder": pipe.transformer.context_embedder,
            "time_text_embed":  pipe.transformer.time_text_embed,
        }

        # MMDiT joint-stream blocks (kept as a reference; not used by current modes)
        self.transformer_blocks = list(pipe.transformer.transformer_blocks)
        n_blocks = len(self.transformer_blocks)

        summary = {
            "pincer_v2": (
                f"  - STYLE recipe (single direction over time):\n"
                f"  - Pooled (2048d) -> pre-hook on time_text_embed (low/zero beta)\n"
                f"  - Sequence (333 tok x caption_projection_dim) -> output hook with single mean direction\n"
                f"    First 77 tokens = CLIP-L+G; next 256 = T5\n"
                f"  - Steers only conditional batch (out[1:]) under CFG\n"
                f"  - {n_blocks} MMDiT blocks present (not directly hooked)"
            ),
            "pincer_perstep": (
                f"  - OBJECT recipe (per-step directions):\n"
                f"  - Pooled (2048d) -> pre-hook on time_text_embed (high beta)\n"
                f"  - Sequence (333 tok x caption_projection_dim) -> per-step output hooks ({self.n_steps} directions)\n"
                f"    First 77 tokens = CLIP-L+G; next 256 = T5\n"
                f"  - Steers only conditional batch (out[1:]) under CFG\n"
                f"  - Total vectors: pooled (1) + sequence ({self.n_steps})"
            ),
        }
        print(f"SD35Steering initialized (mode={mode}, CFG={guidance_scale}):")
        print(summary[mode])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _on_step_end(self, pipe, step, timestep, callback_kwargs):
        self._current_step = int(step.item()) if torch.is_tensor(step) else int(step)
        return callback_kwargs

    def _clear_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _get_clip_l_mask(self, prompt):
        """Mask for the 77-token CLIP-L+G concatenated portion of the sequence."""
        tokenizer = self.pipe.tokenizer
        tok = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt")
        return tok.attention_mask.to(self.device)   # (1, 77)

    def _get_t5_mask(self, prompt):
        """Mask for the 256-token T5 portion of the sequence."""
        tokenizer = self.pipe.tokenizer_3
        tok = tokenizer(prompt, padding="max_length", max_length=256,
                        truncation=True, return_tensors="pt")
        return tok.attention_mask.to(self.device)   # (1, 256)

    def _masked_mean(self, act, mask):
        """Mean over the leading-but-last dims, weighted by mask."""
        mask_f = mask.to(device=act.device, dtype=act.dtype)
        mask_exp = mask_f.unsqueeze(-1)
        weighted = (act * mask_exp).sum(dim=tuple(range(act.dim() - 1)))
        count = mask_exp.sum(dim=tuple(range(act.dim() - 1))).clamp(min=1.0)
        return weighted / count

    def _run_pipe_base(self, prompt, seed, steps=None):
        steps = steps or self.n_steps
        self._current_step = -1
        g = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=self.guidance_scale,
            generator=g,
            callback_on_step_end=self._on_step_end,
        ).images[0]

    # ==================================================================
    # LEARN VECTORS -- diverse-pair (CASteer methodology)
    # ==================================================================
    @torch.no_grad()
    def learn_vectors_diverse(self, prompt_pairs, seed=0, top_k=15, verbose=True):
        """
        Capture per-step pooled + sequence directions over diverse pairs.

        For each (positive, negative) prompt pair (CASteer Appendix C format):
          * Run the pipeline once for positive, once for negative (same seed).
          * Hook context_embedder OUTPUT to record the (333, D) sequence
            activation at every denoising step, where D = caption_projection_dim
            (1536 for SD3.5-medium, 2432 for SD3.5-large). Only the CONDITIONAL
            position (batch index 1 under CFG) is recorded.
          * Hook time_text_embed PRE to record the 2048d pooled embedding
            (conditional position only).
        After all N pairs:
          * pooled direction: mean(pos_pooled - neg_pooled) over pairs, normed
          * pincer_v2 (style): single time-averaged sequence direction (mean over
            all step indices), CLIP-section and T5-section pooled separately
            via masked mean.
          * pincer_perstep (object): per-step sequence directions, CLIP- and
            T5-section directions per step.

        Returns dict matching FLUX's convention but with SD3.5 keys:
          {
            "pooled_2048": {0: direction(2048,)},
            "ctx_clip":    {step: direction(caption_projection_dim,)},   # over CLIP 77-token region
            "ctx_t5":      {step: direction(caption_projection_dim,)},   # over T5 256-token region
          }
        """
        n_pairs = len(prompt_pairs)
        if verbose:
            print(f"Learning from {n_pairs} diverse pairs (seed={seed}, mode={self.mode})")

        # Per-step accumulators for sequence directions
        # CLIP region (first 77 tokens) and T5 region (next 256) handled separately
        # because they have different attention masks.
        ctx_clip_acc = {step: None for step in range(self.n_steps)}
        ctx_t5_acc   = {step: None for step in range(self.n_steps)}
        # Pooled accumulator (single direction, all steps share)
        pooled_acc = None

        # Inner-pass capture state
        cap_ctx_seq = {}    # {step: (333, 3072)}
        cap_pooled  = {}    # {"pooled": (2048,)}

        def _ctx_hook(module, inp, out):
            # out shape: (B=2 under CFG, 333, 3072)
            step = self._current_step + 1
            if 0 <= step < self.n_steps:
                # Take conditional position only (index 1 under CFG)
                cap_ctx_seq[step] = out[1].detach().float().cpu()

        def _pooled_pre_hook(module, args):
            # args = (timestep, pooled_projections); pooled is (B=2, 2048)
            if "pooled" not in cap_pooled:
                pooled = args[1]
                cap_pooled["pooled"] = pooled[1].detach().float().cpu()   # cond only

        def _capture(prompt, seed_):
            cap_ctx_seq.clear()
            cap_pooled.clear()
            self._clear_hooks()
            self._handles.append(
                self.target_layers["context_embedder"].register_forward_hook(_ctx_hook))
            self._handles.append(
                self.target_layers["time_text_embed"].register_forward_pre_hook(_pooled_pre_hook))
            try:
                self._run_pipe_base(prompt, seed_)
            finally:
                self._clear_hooks()
            return dict(cap_ctx_seq), cap_pooled.get("pooled")

        for pair_idx, (pos_p, neg_p) in enumerate(
            tqdm(prompt_pairs, desc="Diverse pairs", disable=not verbose)
        ):
            pos_clip_mask = self._get_clip_l_mask(pos_p)
            pos_t5_mask   = self._get_t5_mask(pos_p)
            neg_clip_mask = self._get_clip_l_mask(neg_p)
            neg_t5_mask   = self._get_t5_mask(neg_p)

            pos_ctx, pos_pooled = _capture(pos_p, seed)
            neg_ctx, neg_pooled = _capture(neg_p, seed)

            for step in range(self.n_steps):
                if step not in pos_ctx or step not in neg_ctx:
                    continue
                pos_seq = pos_ctx[step]   # (333, 3072)
                neg_seq = neg_ctx[step]
                # Pad up if shapes differ slightly (should not happen, but safe)
                seq_len = min(pos_seq.shape[0], neg_seq.shape[0])
                pos_seq = pos_seq[:seq_len]
                neg_seq = neg_seq[:seq_len]

                # CLIP region: first 77 tokens. Use positive prompt's CLIP mask.
                clip_diff = pos_seq[:77] - neg_seq[:77]   # (77, 3072)
                clip_pos_pool = self._masked_mean(
                    pos_seq[:77].unsqueeze(0), pos_clip_mask)  # (3072,)
                clip_neg_pool = self._masked_mean(
                    neg_seq[:77].unsqueeze(0), neg_clip_mask)
                d_clip = clip_pos_pool - clip_neg_pool   # (3072,)
                ctx_clip_acc[step] = d_clip if ctx_clip_acc[step] is None else ctx_clip_acc[step] + d_clip

                # T5 region: tokens 77 onwards (up to 77+256=333).
                t5_pos_pool = self._masked_mean(
                    pos_seq[77:77 + 256].unsqueeze(0), pos_t5_mask)
                t5_neg_pool = self._masked_mean(
                    neg_seq[77:77 + 256].unsqueeze(0), neg_t5_mask)
                d_t5 = t5_pos_pool - t5_neg_pool
                ctx_t5_acc[step] = d_t5 if ctx_t5_acc[step] is None else ctx_t5_acc[step] + d_t5

            if pos_pooled is not None and neg_pooled is not None:
                d_pool = pos_pooled - neg_pooled   # (2048,)
                pooled_acc = d_pool if pooled_acc is None else pooled_acc + d_pool

            if verbose and (pair_idx + 1) % 10 == 0:
                print(f"  Completed {pair_idx + 1}/{n_pairs} prompt pairs")

        # Build output vectors dict
        vectors = {}

        # Pooled direction (single, all steps share)
        if pooled_acc is not None:
            avg = pooled_acc / n_pairs
            direction = avg / (avg.norm() + 1e-8)
            vectors["pooled_2048"] = {0: direction.to(self.device, dtype=DTYPE)}

        # Sequence directions
        if self.mode == "pincer_v2":
            # Single time-averaged direction (mean over steps)
            clip_avg = sum(v for v in ctx_clip_acc.values() if v is not None) / max(
                1, sum(1 for v in ctx_clip_acc.values() if v is not None))
            clip_avg = clip_avg / n_pairs
            t5_avg = sum(v for v in ctx_t5_acc.values() if v is not None) / max(
                1, sum(1 for v in ctx_t5_acc.values() if v is not None))
            t5_avg = t5_avg / n_pairs
            vectors["ctx_clip"] = {0: (clip_avg / (clip_avg.norm() + 1e-8)).to(
                self.device, dtype=DTYPE)}
            vectors["ctx_t5"]   = {0: (t5_avg / (t5_avg.norm() + 1e-8)).to(
                self.device, dtype=DTYPE)}
        else:  # pincer_perstep
            ctx_clip_dirs = {}
            ctx_t5_dirs = {}
            for step in range(self.n_steps):
                if ctx_clip_acc[step] is not None:
                    avg = ctx_clip_acc[step] / n_pairs
                    ctx_clip_dirs[step] = (avg / (avg.norm() + 1e-8)).to(
                        self.device, dtype=DTYPE)
                if ctx_t5_acc[step] is not None:
                    avg = ctx_t5_acc[step] / n_pairs
                    ctx_t5_dirs[step] = (avg / (avg.norm() + 1e-8)).to(
                        self.device, dtype=DTYPE)
            vectors["ctx_clip"] = ctx_clip_dirs
            vectors["ctx_t5"]   = ctx_t5_dirs

        if verbose:
            print(f"\n{'='*70}")
            print(f"SD3.5 Steering Vectors ({self.mode}, {n_pairs} pairs)")
            print(f"{'='*70}")
            print(f"{'Component':<20} {'#Steps':<10} {'Sample norm':<15}")
            print(f"{'-'*70}")
            for k, step_vecs in vectors.items():
                sample = next(iter(step_vecs.values()))
                print(f"{k:<20} {len(step_vecs):<10} {float(sample.norm()):<15.4f}")
            print(f"{'='*70}\n")

        return vectors

    # ==================================================================
    # APPLY VECTORS -- CFG-aware steering
    # ==================================================================
    @contextmanager
    def apply_vectors(self, vectors, beta=2.0, clip_negative=True,
                      step_range=None, clip_cap=None):
        """
        Apply steering vectors during generation.

        Hook points:
          * "pooled_2048" -> pre-hook on time_text_embed input. Only modifies
            args[1][1] (conditional pooled), leaving args[1][0] (uncond) intact.
          * "ctx_clip"  -> output hook on context_embedder, modifies first 77
            tokens of out[1] only (conditional, CLIP region).
          * "ctx_t5"    -> output hook on context_embedder, modifies tokens
            77..333 of out[1] only (conditional, T5 region).

        Args:
          vectors: dict from learn_vectors_diverse.
          beta: float or dict {"pooled": .., "clip": .., "t5": ..}.
          clip_negative: clamp projection score to [0, inf) before subtracting.
            Recommended True; disables "negative projection adds concept" failure.
          step_range: (start, end) tuple of denoising steps to fire on.
          clip_cap: ceiling on effective β for the pooled hook.
            None = no cap (object recipe). 1.0 = style guardrail.
        """
        if isinstance(beta, dict):
            beta_pool = beta.get("pooled", 5.0)
            beta_clip = beta.get("clip", 5.0)
            beta_t5   = beta.get("t5", 5.0)
        else:
            beta_pool = beta_clip = beta_t5 = beta

        if step_range is None:
            def _in_range(step):
                return True
        else:
            s_start, s_end = step_range
            def _in_range(step):
                return s_start <= step < s_end

        # ----- pooled (2048d) pre-hook on time_text_embed -----
        def pooled_pre_hook(direction, b):
            def hook(module, args):
                step = self._current_step + 1
                if not _in_range(step):
                    return None
                timestep, pooled = args
                d = direction.to(pooled.device, pooled.dtype)
                pooled_new = pooled.clone()
                # Modify only the conditional position (index 1 under CFG).
                # If batch size is 1 (no CFG), index 0.
                cond_idx = 1 if pooled.shape[0] >= 2 else 0
                cond = pooled_new[cond_idx]
                score = (cond @ d)
                if clip_negative:
                    score = score.clamp(min=0.0)
                if clip_cap is None:
                    effective = float(b) * score
                else:
                    effective = min(float(b), float(clip_cap)) * score
                update = effective * d
                pooled_new[cond_idx] = cond - update
                return (timestep, pooled_new)
            return hook

        # ----- sequence (3072d) output hook on context_embedder -----
        # Two regions: CLIP (first 77) and T5 (next 256).
        # Modifies only conditional position.
        def ctx_output_hook(layer_clip_vecs, layer_t5_vecs, b_clip_local, b_t5_local):
            def hook(module, inputs, output):
                step = self._current_step + 1
                if not _in_range(step):
                    return output
                cond_idx = 1 if output.shape[0] >= 2 else 0
                out = output.clone()

                # CLIP region: first 77 tokens
                d_clip = layer_clip_vecs.get(step)
                if d_clip is None:
                    d_clip = layer_clip_vecs.get(0)
                if d_clip is not None:
                    d = d_clip.to(out.device, out.dtype)
                    cond_clip = out[cond_idx, :77]   # (77, D)
                    score = cond_clip @ d            # (77,)
                    if clip_negative:
                        score = score.clamp(min=0.0)
                    update = (b_clip_local * score).unsqueeze(-1) * d  # (77, D)
                    out[cond_idx, :77] = cond_clip - update

                # T5 region: tokens 77..333
                d_t5 = layer_t5_vecs.get(step)
                if d_t5 is None:
                    d_t5 = layer_t5_vecs.get(0)
                if d_t5 is not None:
                    d = d_t5.to(out.device, out.dtype)
                    cond_t5 = out[cond_idx, 77:333]
                    score = cond_t5 @ d
                    if clip_negative:
                        score = score.clamp(min=0.0)
                    update = (b_t5_local * score).unsqueeze(-1) * d
                    out[cond_idx, 77:333] = cond_t5 - update

                return out
            return hook

        try:
            self._clear_hooks()

            # Pooled hook (always added if pooled_2048 present)
            pooled_vecs = vectors.get("pooled_2048")
            if pooled_vecs is not None and 0 in pooled_vecs and beta_pool > 0:
                self._handles.append(
                    self.target_layers["time_text_embed"].register_forward_pre_hook(
                        pooled_pre_hook(pooled_vecs[0], beta_pool)))

            # Sequence hook (combined for CLIP + T5 regions)
            ctx_clip_vecs = vectors.get("ctx_clip", {})
            ctx_t5_vecs   = vectors.get("ctx_t5", {})
            if ctx_clip_vecs or ctx_t5_vecs:
                self._handles.append(
                    self.target_layers["context_embedder"].register_forward_hook(
                        ctx_output_hook(ctx_clip_vecs, ctx_t5_vecs,
                                        beta_clip, beta_t5)))
            yield
        finally:
            self._clear_hooks()

    # ==================================================================
    # GENERATE
    # ==================================================================
    def generate(self, prompt, seed, vectors=None, beta=2.0, clip_negative=True,
                 step_range=None, clip_cap=None):
        if vectors:
            with self.apply_vectors(vectors, beta=beta, clip_negative=clip_negative,
                                    step_range=step_range, clip_cap=clip_cap):
                return self._run_pipe_base(prompt, seed)
        else:
            return self._run_pipe_base(prompt, seed)

    # ==================================================================
    # SAVE / LOAD
    # ==================================================================
    def save_vectors(self, vectors, filepath):
        save = {}
        for k, step_dict in vectors.items():
            save[k] = {step: t.cpu() for step, t in step_dict.items()}
        torch.save(save, filepath)
        print(f"Saved steering vectors to: {filepath}")

    def load_vectors(self, filepath):
        save = torch.load(filepath, map_location=self.device)
        return {k: {step: t.to(self.device, dtype=DTYPE) for step, t in v.items()}
                for k, v in save.items()}


print("✓ SD35Steering class defined!")

# ============================================================================
# CELL 4: QUALITY METRICS (FID, CLIP Score)
# ============================================================================

class QualityMetrics:
    """Calculate image quality metrics for UnlearnCanvas evaluation."""

    def __init__(self, device="cuda"):
        self.device = device
        print("Loading quality metric models...")
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
            self.clip_model.eval()
            print("  ✓ CLIP ViT-L/14 loaded")
        except Exception as e:
            print(f"  ✗ CLIP loading failed: {e}")
            self.clip_model = None

    def calculate_clip_score(self, images, prompts):
        if self.clip_model is None:
            return None
        scores = []
        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                text_input = clip.tokenize([prompt], truncate=True).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores.append((image_features @ text_features.T).item())
        return float(np.mean(scores))

    def calculate_fid(self, real_path, generated_path):
        try:
            return fid.compute_fid(
                real_path, generated_path, mode="clean",
                num_workers=0, batch_size=8,
                device=torch.device(self.device))
        except Exception as e:
            print(f"⚠ FID calculation error: {e}")
            return None


print("✓ QualityMetrics class defined!")

# ============================================================================
# CELL 4B: LLAVA CLASSIFIER (TRACE format)
# ============================================================================

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class LLaVAClassifier:
    """LLaVA-1.6-Vicuna-7B classifier (TRACE Figures 6/7 numbered-options format)."""

    def __init__(self, model_id="llava-hf/llava-v1.6-vicuna-7b-hf", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        print(f"Loading LLaVA: {self.model_id}...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto")
        print("✓ LLaVA loaded!")

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            print("✓ LLaVA unloaded")

    def _generate_response(self, image, prompt):
        if self.model is None:
            self.load()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        conv = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": prompt}]}]
        prompt_formatted = self.processor.apply_chat_template(
            conv, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt_formatted,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        return self.processor.decode(
            out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    def _parse_number(self, response, max_options):
        import re
        nums = re.findall(r'\d+', response)
        if nums:
            n = int(nums[0])
            if 1 <= n <= max_options:
                return n - 1
        return None

    def classify_style(self, image, styles=None, debug=False):
        styles = styles or STYLES
        opts = '\n'.join([f"{i+1}. {s.replace('_', ' ')}" for i, s in enumerate(styles)])
        prompt = (
            "You are an image classifier. Classify the artistic style of the given image.\n"
            "Instruction: Choose exactly one option from the numbered list below. "
            "Respond with only the number.\n"
            f"Options:\n{opts}")
        resp = self._generate_response(image, prompt)
        if debug:
            print(f"LLaVA style: '{resp}'")
        idx = self._parse_number(resp, len(styles))
        if idx is not None:
            return styles[idx]
        rl = resp.lower()
        for s in styles:
            if s.lower().replace('_', ' ') in rl:
                return s
        return None

    def classify_object(self, image, objects=None, debug=False):
        objects = objects or OBJECTS
        opts = '\n'.join([f"{i+1}. {o.replace('_', ' ')}" for i, o in enumerate(objects)])
        prompt = (
            "Classify the object depicted in this image.\n"
            "Choose exactly one option from the numbered list.\n"
            "Respond with only the number.\n"
            f"Object categories:\n{opts}")
        resp = self._generate_response(image, prompt)
        if debug:
            print(f"LLaVA object: '{resp}'")
        idx = self._parse_number(resp, len(objects))
        if idx is not None:
            return objects[idx]
        rl = resp.lower()
        for o in objects:
            if o.lower().replace('_', ' ') in rl:
                return o
        return None


print("✓ LLaVAClassifier class defined!")

# ============================================================================
# CELL 5: UNLEARNCANVAS EVALUATOR
# ============================================================================

class UnlearnCanvasEvaluator:
    """UA / IRA / CRA evaluation with TRACE-correct scoring (CRA on non-target only)."""

    def __init__(self, device="cuda"):
        self.device = device
        self.llava = LLaVAClassifier(device=device)
        print("✓ UnlearnCanvas Evaluator ready (LLaVA-1.6-Vicuna-7B)")

    def classify_image(self, image, domain="style"):
        if domain == "style":
            return self.llava.classify_style(image)
        else:
            return self.llava.classify_object(image)

    def evaluate_unlearning(
        self, steerer, vectors, target_concept, target_type="object",
        beta=2.0, clip_negative=True, step_range=None, clip_cap=None,
        eval_seeds=None, save_images=True, output_dir=None,
        generate_baselines=False
    ):
        eval_seeds = eval_seeds or EVAL_SEEDS
        output_dir = output_dir or os.path.join(
            STEERED_DIR, f"{target_concept}_{steerer.mode}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"EVALUATING UNLEARNING: {target_concept} ({target_type})")
        print(f"{'=' * 70}")

        # Build full grid
        test_cases = []
        for style in STYLES:
            for obj in OBJECTS:
                for seed in eval_seeds:
                    fname = f"{style}_{obj}_seed{seed}.jpg"
                    prompt = f"A {obj} image in {style.replace('_', ' ')} style."
                    test_cases.append({
                        "prompt": prompt, "seed": seed,
                        "gt_style": style, "gt_object": obj,
                        "filename": fname
                    })
        total_images = len(test_cases)

        # Phase 1: generation
        skipped = generated = 0
        print(f"\n--- PHASE 1: IMAGE GENERATION ---")
        print(f"Grid: {len(STYLES)} styles x {len(OBJECTS)} objects x "
              f"{len(eval_seeds)} seeds = {total_images} images")
        print(f"Steering: beta={beta}, step_range={step_range}, clip_cap={clip_cap}")
        print(f"Output: {output_dir}")

        for i, case in enumerate(tqdm(test_cases, desc="Phase 1: Generating")):
            save_path = os.path.join(output_dir, case["filename"])
            if os.path.exists(save_path):
                skipped += 1
                continue
            img = steerer.generate(
                case["prompt"], case["seed"], vectors=vectors,
                beta=beta, clip_negative=clip_negative,
                step_range=step_range, clip_cap=clip_cap)
            img.save(save_path)
            generated += 1
        print(f"Phase 1 done: {generated} generated, {skipped} skipped (existed)")

        # Free SD3.5 VRAM before LLaVA
        print("\nFreeing SD3.5 VRAM before classification phase...")
        steerer.pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 2: classification (TRACE-correct scoring)
        print(f"\n--- PHASE 2: CLASSIFICATION ({total_images} images) ---")
        results = {
            "target_correct": 0, "target_total": 0,
            "ira_correct": 0,    "ira_total": 0,
            "cra_correct": 0,    "cra_total": 0,
            "prompts": []
        }

        for i, case in enumerate(tqdm(test_cases, desc="Phase 2: Classifying")):
            img_path = os.path.join(output_dir, case["filename"])
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            results["prompts"].append(case["prompt"])
            gt_style = case["gt_style"]
            gt_object = case["gt_object"]

            if target_type == "style":
                pred_style = self.classify_image(img, domain="style")
                pred_object = self.classify_image(img, domain="object")
                if gt_style == target_concept:
                    results["target_total"] += 1
                    if pred_style == target_concept:
                        results["target_correct"] += 1
                else:
                    results["ira_total"] += 1
                    if pred_style == gt_style:
                        results["ira_correct"] += 1
                    results["cra_total"] += 1
                    if pred_object == gt_object:
                        results["cra_correct"] += 1
            else:   # object target
                pred_object = self.classify_image(img, domain="object")
                pred_style = self.classify_image(img, domain="style")
                if gt_object == target_concept:
                    results["target_total"] += 1
                    if pred_object == target_concept:
                        results["target_correct"] += 1
                else:
                    results["ira_total"] += 1
                    if pred_object == gt_object:
                        results["ira_correct"] += 1
                    results["cra_total"] += 1
                    if pred_style == gt_style:
                        results["cra_correct"] += 1

            if (i + 1) % 50 == 0:
                _ua = 1.0 - (results["target_correct"] / max(results["target_total"], 1))
                _ira = results["ira_correct"] / max(results["ira_total"], 1)
                _cra = results["cra_correct"] / max(results["cra_total"], 1)
                print(f"  [{i+1}/{total_images}] running UA={_ua:.1%} IRA={_ira:.1%} CRA={_cra:.1%}")

        # Reload SD3.5 to GPU
        print("\nClassification done. Unloading LLaVA, reloading SD3.5 to GPU...")
        self.llava.unload()
        steerer.pipe.to(steerer.device)

        ua = 1.0 - (results["target_correct"] / max(results["target_total"], 1))
        ira = results["ira_correct"] / max(results["ira_total"], 1)
        cra = results["cra_correct"] / max(results["cra_total"], 1)

        print(f"\n{'=' * 70}")
        print(f"FINAL RESULTS: {target_concept}")
        print(f"{'=' * 70}")
        print(f"UA  : {ua:.2%}")
        print(f"IRA : {ira:.2%}")
        print(f"CRA : {cra:.2%}")
        print(f"{'=' * 70}")

        return {
            "UA": ua, "IRA": ira, "CRA": cra,
            "target_concept": target_concept, "target_type": target_type,
            "beta": beta, "n_images": total_images, "prompts": results["prompts"]
        }


print("✓ UnlearnCanvasEvaluator class defined!")

# ============================================================================
# CELL 6: LOAD MODELS
# ============================================================================

print("\nLoading SD3.5 pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
print(f"✓ SD3.5 pipeline loaded ({MODEL_ID})")

# ----------------------------------------------------------------------
# Architecture probe -- assert the layout this file's hooks rely on.
# Verified against diffusers main branch (Nov 2024):
#   src/diffusers/models/transformers/transformer_sd3.py
#   src/diffusers/models/embeddings.py (CombinedTimestepTextProjEmbeddings)
#   src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# ----------------------------------------------------------------------
print("\nArchitecture probe:")
_t = pipe.transformer
_caption_dim = _t.config.caption_projection_dim
_n_layers = len(_t.transformer_blocks)
_pooled_dim = _t.config.pooled_projection_dim
print(f"  caption_projection_dim : {_caption_dim}     "
      f"(expected: 1536 medium / 2432 large)")
print(f"  pooled_projection_dim  : {_pooled_dim}     "
      f"(expected: 2048 = CLIP-L 768 + CLIP-G 1280)")
print(f"  num_transformer_blocks : {_n_layers}        "
      f"(expected: 24 medium / 38 large)")
print(f"  context_embedder       : {type(_t.context_embedder).__name__}  "
      f"(expected: Linear)")
print(f"  time_text_embed        : {type(_t.time_text_embed).__name__}  "
      f"(expected: CombinedTimestepTextProjEmbeddings)")

assert _pooled_dim == 2048, (
    f"pooled_projection_dim={_pooled_dim} != 2048; the pooled hook assumes "
    f"a 2048-d concatenated [CLIP-L||CLIP-G] vector. Aborting before "
    f"running steering.")
assert isinstance(_t.context_embedder, torch.nn.Linear), (
    f"context_embedder is {type(_t.context_embedder).__name__}, expected "
    f"nn.Linear. The output-hook math assumes a single Linear projection.")
assert _t.context_embedder.in_features == 4096, (
    f"context_embedder.in_features={_t.context_embedder.in_features} != 4096; "
    f"this code assumes joint_attention_dim=4096 as input.")
assert _t.context_embedder.out_features == _caption_dim, (
    f"context_embedder.out_features={_t.context_embedder.out_features} != "
    f"caption_projection_dim={_caption_dim}.")
print(f"  ✓ all architectural assumptions hold")

# ===== EXPERIMENT TARGET =====
TARGET_CONCEPT = "Dogs"        # plural form to match TRACE
TARGET_TYPE    = "object"      # "style" or "object"

# Mode auto-selection (parallel to FLUX)
if TARGET_TYPE == "style":
    STEERING_MODE = "pincer_v2"
else:
    STEERING_MODE = "pincer_perstep"

print(f"\nInitializing SD35Steering (mode={STEERING_MODE}, CFG={GUIDANCE_SCALE})...")
steerer = SD35Steering(
    pipe, device=DEVICE, n_steps=N_STEPS,
    mode=STEERING_MODE, guidance_scale=GUIDANCE_SCALE)

print("\nInitializing evaluators...")
evaluator = UnlearnCanvasEvaluator(device=DEVICE)
quality_metrics = QualityMetrics(device=DEVICE)

print("\n" + "=" * 70)
print("✓ ALL MODELS LOADED")
print("=" * 70)

# ============================================================================
# CELL 7: EXPERIMENT CONFIGURATION
# ============================================================================
"""
Per-mode default hyperparameters. Cell 12B's per-concept search overrides
these with concept-specific values; this block is the global fallback.

Calibration vs FLUX:
  FLUX-schnell runs at CFG=0 (distilled), so β=3-5 suffices.
  SD3.5 runs at CFG=4-7. CFG amplifies the conditional pathway by w=4..7,
  so β has to be modestly higher to overcome that amplification, but NOT
  100-2000x as the previous file required (that was due to CFG=7 + only
  25 pairs + steering both CFG branches). Reasonable starting range
  here: β in [3, 15] for both pooled and ctx components.
"""
if TARGET_TYPE == "style":
    BETA = {"pooled": 0.0, "clip": 5.0, "t5": 5.0}
    STEP_RANGE = (0, N_STEPS)
    CLIP_CAP = 1.0    # style guardrail
else:
    BETA = {"pooled": 8.0, "clip": 5.0, "t5": 8.0}
    STEP_RANGE = (0, N_STEPS)
    CLIP_CAP = None   # object: no cap so β > 1 can push past zero

# Output directory includes mode to prevent cross-mode caching
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"{TARGET_CONCEPT}_{STEERING_MODE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Diverse prompt pairs (CASteer Appendix C format)
if TARGET_TYPE == "style":
    DIVERSE_PROMPT_PAIRS = make_style_prompts(
        TARGET_CONCEPT.replace('_', ' '), NUM_DIVERSE_PROMPTS)
else:
    DIVERSE_PROMPT_PAIRS = make_object_prompts(
        TARGET_CONCEPT.replace('_', ' '), NUM_DIVERSE_PROMPTS)

print("=" * 70)
print("EXPERIMENT CONFIGURATION (SD3.5)")
print("=" * 70)
print(f"Target Concept:    {TARGET_CONCEPT}")
print(f"Target Type:       {TARGET_TYPE}")
print(f"Steering Mode:     {STEERING_MODE}")
print(f"Steering β:        {BETA}")
print(f"Step range:        {STEP_RANGE}")
print(f"CLIP cap:          {CLIP_CAP}")
print(f"Guidance scale:    {GUIDANCE_SCALE}")
print(f"Diverse pairs:     {len(DIVERSE_PROMPT_PAIRS)}")
print(f"  Example pos:     '{DIVERSE_PROMPT_PAIRS[0][0]}'")
print(f"  Example neg:     '{DIVERSE_PROMPT_PAIRS[0][1]}'")
print(f"Eval seeds:        {len(EVAL_SEEDS)}")
print(f"Output:            {OUTPUT_DIR}")
print("=" * 70)

# ============================================================================
# CELL 8: LEARN STEERING VECTORS
# ============================================================================

print(f"\nLearning steering vectors for: {TARGET_CONCEPT} ({STEERING_MODE})")
vector_path = os.path.join(
    VECTOR_DIR, f"{TARGET_CONCEPT}_{STEERING_MODE}_diverse_vectors.pt")
if os.path.exists(vector_path):
    print(f"Loading cached vectors from {vector_path}")
    vectors = steerer.load_vectors(vector_path)
else:
    vectors = steerer.learn_vectors_diverse(
        prompt_pairs=DIVERSE_PROMPT_PAIRS, seed=0,
        top_k=TOP_K_VECTORS, verbose=True)
    steerer.save_vectors(vectors, vector_path)

# ============================================================================
# CELL 8B: QUICK STEERING TEST
# ============================================================================
"""
Generates one baseline + a small β sweep so you can confirm steering
actually does something before kicking off the full eval.
"""

DIAG_PROMPT = f"A {TARGET_CONCEPT.replace('_', ' ')} image in Van Gogh style."
DIAG_SEED = 188

print("=" * 70)
print(f"QUICK STEERING TEST: {TARGET_CONCEPT} ({STEERING_MODE})")
print("=" * 70)
print(f"Prompt: '{DIAG_PROMPT}'")

print("\nGenerating baseline (no steering)...")
baseline_img = steerer.generate(DIAG_PROMPT, DIAG_SEED, vectors=None)

if STEERING_MODE == "pincer_perstep":
    configs = [
        ("pool=3,clip=3,t5=5",  {"pooled": 3.0, "clip": 3.0, "t5": 5.0}, True, (0, N_STEPS), None),
        ("pool=8,clip=5,t5=8",  {"pooled": 8.0, "clip": 5.0, "t5": 8.0}, True, (0, N_STEPS), None),
        ("pool=15,clip=8,t5=12",{"pooled": 15.0, "clip": 8.0, "t5": 12.0}, True, (0, N_STEPS), None),
        ("pool=25,clip=10,t5=15",{"pooled": 25.0, "clip": 10.0, "t5": 15.0}, True, (0, N_STEPS), None),
    ]
else:
    configs = [
        ("clip=3,t5=3", {"pooled": 0.0, "clip": 3.0, "t5": 3.0}, True, (0, N_STEPS), 1.0),
        ("clip=5,t5=5", {"pooled": 0.0, "clip": 5.0, "t5": 5.0}, True, (0, N_STEPS), 1.0),
        ("clip=8,t5=8", {"pooled": 0.0, "clip": 8.0, "t5": 8.0}, True, (0, N_STEPS), 1.0),
        ("clip=12,t5=12",{"pooled": 0.0, "clip": 12.0, "t5": 12.0}, True, (0, N_STEPS), 1.0),
    ]

test_images = []
for label, beta_val, clip_neg, srng, ccap in configs:
    print(f"  Generating: {label}...")
    img = steerer.generate(DIAG_PROMPT, DIAG_SEED, vectors=vectors,
                           beta=beta_val, clip_negative=clip_neg,
                           step_range=srng, clip_cap=ccap)
    test_images.append((label, img))

baseline_arr = np.array(baseline_img).astype(float)
fig, axes = plt.subplots(1, 1 + len(test_images), figsize=(4 * (1 + len(test_images)), 4))
axes[0].imshow(baseline_img)
axes[0].set_title("Baseline\n(no steering)", fontsize=10)
axes[0].axis("off")
for i, (label, img) in enumerate(test_images):
    diff = np.abs(np.array(img).astype(float) - baseline_arr)
    pct = (diff > 1.0).mean() * 100
    axes[i + 1].imshow(img)
    axes[i + 1].set_title(f"{label}\n{pct:.0f}% changed", fontsize=10,
                          color='green' if pct > 10 else 'orange' if pct > 1 else 'red')
    axes[i + 1].axis("off")
plt.suptitle(f"SD3.5 Steering Test: {TARGET_CONCEPT} ({STEERING_MODE})", fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 9: SINGLE-TARGET FULL UNLEARNCANVAS EVALUATION (UA/IRA/CRA)
# ============================================================================

CLIP_NEGATIVE = True
print("\nRunning single-target UnlearnCanvas evaluation...")
eval_results = evaluator.evaluate_unlearning(
    steerer=steerer, vectors=vectors,
    target_concept=TARGET_CONCEPT, target_type=TARGET_TYPE,
    beta=BETA, clip_negative=CLIP_NEGATIVE,
    step_range=STEP_RANGE, clip_cap=CLIP_CAP,
    eval_seeds=EVAL_SEEDS, save_images=True, output_dir=OUTPUT_DIR,
    generate_baselines=False)

print("\n" + "=" * 70)
print("SINGLE-TARGET RESULTS")
print("=" * 70)
print(f"{TARGET_CONCEPT}: UA={eval_results['UA']:.2%}  IRA={eval_results['IRA']:.2%}  CRA={eval_results['CRA']:.2%}")

# ============================================================================
# CELL 12B: PER-CONCEPT HYPERPARAMETER SEARCH
# ============================================================================
"""
Per-concept beta search before the full sweep. Mirrors FLUX's Cell 12B
exactly so paper methodology is identical across models. Cached to Drive
(skipped on re-run). Composite score = 0.5*UA + 0.25*IRA + 0.25*CRA.
"""
import json as _json

FORCE_HPARAM_SEARCH = False

best_params_path = os.path.join(
    TABLES_DIR, f"best_params_{TARGET_TYPE}_{STEERING_MODE}.json")
search_log_path = os.path.join(
    TABLES_DIR, f"search_log_{TARGET_TYPE}_{STEERING_MODE}.json")

if TARGET_TYPE == "style":
    SEARCH_BETAS = [
        {"pooled": 0.0, "clip": 3.0, "t5": 3.0},
        {"pooled": 0.0, "clip": 5.0, "t5": 5.0},
        {"pooled": 0.0, "clip": 8.0, "t5": 8.0},
        {"pooled": 0.0, "clip": 12.0, "t5": 12.0},
    ]
    SEARCH_STEP_RANGE = (0, N_STEPS)
    SEARCH_CLIP_NEG = True
    SEARCH_CLIP_CAP = 1.0
else:
    SEARCH_BETAS = [
        {"pooled": 3.0,  "clip": 3.0, "t5": 5.0},
        {"pooled": 8.0,  "clip": 5.0, "t5": 8.0},
        {"pooled": 15.0, "clip": 8.0, "t5": 12.0},
        {"pooled": 25.0, "clip": 10.0, "t5": 15.0},
        {"pooled": 40.0, "clip": 12.0, "t5": 20.0},
    ]
    SEARCH_STEP_RANGE = (0, N_STEPS)
    SEARCH_CLIP_NEG = True
    SEARCH_CLIP_CAP = None

PROXY_OBJECTS_BASE = ["Dogs", "Cats"]
PROXY_STYLES_BASE  = ["Van_Gogh", "Cartoon", "Watercolor"]
PROXY_SEEDS        = [188, 288]

def _build_proxy_for_concept(concept, target_type):
    if target_type == "style":
        styles = list(dict.fromkeys([concept] + PROXY_STYLES_BASE))[:3]
        objects = PROXY_OBJECTS_BASE
    else:
        styles = PROXY_STYLES_BASE
        objects = list(dict.fromkeys([concept] + PROXY_OBJECTS_BASE))[:2]
    return styles, objects

PROXY_DIR = os.path.join(
    STEERED_DIR, f"_hparam_proxy_{TARGET_TYPE}_{STEERING_MODE}")
os.makedirs(PROXY_DIR, exist_ok=True)

def _composite_score(ua, ira, cra, target_type=None):
    return 0.5 * ua + 0.25 * ira + 0.25 * cra

if os.path.exists(best_params_path) and not FORCE_HPARAM_SEARCH:
    with open(best_params_path) as f:
        best_params = _json.load(f)
    print(f"Loaded cached best_params from {best_params_path}")
    print(f"  ({len(best_params)} concepts cached)")
else:
    if TARGET_TYPE == "style":
        SEARCH_CONCEPTS = STYLES
        make_pairs_search = make_style_prompts
    else:
        SEARCH_CONCEPTS = OBJECTS
        make_pairs_search = make_object_prompts

    print(f"\n{'=' * 70}")
    print(f"HYPERPARAMETER SEARCH: {TARGET_TYPE.upper()} ({STEERING_MODE})")
    print(f"{'=' * 70}")
    print(f"Concepts:    {len(SEARCH_CONCEPTS)}")
    print(f"Beta combos: {len(SEARCH_BETAS)}")
    print(f"Proxy/combo: ~12 images")

    # Phase 1: generate proxy images
    print(f"\n--- PHASE 1: PROXY GENERATION ---")
    steerer.pipe.to(steerer.device)
    for c_idx, concept in enumerate(SEARCH_CONCEPTS):
        print(f"\n  [{c_idx + 1}/{len(SEARCH_CONCEPTS)}] {concept}")
        vpath = os.path.join(
            VECTOR_DIR, f"{concept}_{STEERING_MODE}_diverse_vectors.pt")
        if os.path.exists(vpath):
            vectors = steerer.load_vectors(vpath)
        else:
            pairs = make_pairs_search(concept.replace('_', ' '), NUM_DIVERSE_PROMPTS)
            vectors = steerer.learn_vectors_diverse(
                prompt_pairs=pairs, seed=0, top_k=TOP_K_VECTORS, verbose=False)
            steerer.save_vectors(vectors, vpath)

        proxy_styles, proxy_objects = _build_proxy_for_concept(concept, TARGET_TYPE)
        for beta_dict in tqdm(SEARCH_BETAS, desc="  combos", leave=False):
            combo_key = (f"pool{beta_dict['pooled']}_"
                         f"clip{beta_dict['clip']}_t5{beta_dict['t5']}")
            for style in proxy_styles:
                for obj in proxy_objects:
                    for seed in PROXY_SEEDS:
                        fname = f"{concept}__{combo_key}__{style}_{obj}_seed{seed}.jpg"
                        fp = os.path.join(PROXY_DIR, fname)
                        if os.path.exists(fp):
                            continue
                        prompt = f"A {obj} image in {style.replace('_', ' ')} style."
                        img = steerer.generate(
                            prompt, seed, vectors=vectors, beta=beta_dict,
                            clip_negative=SEARCH_CLIP_NEG,
                            step_range=SEARCH_STEP_RANGE,
                            clip_cap=SEARCH_CLIP_CAP)
                        img.save(fp)
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2: LLaVA scoring
    print(f"\n--- PHASE 2: LLAVA SCORING ---")
    steerer.pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    evaluator.llava.load()

    search_results = {}
    best_params = {}
    for c_idx, concept in enumerate(SEARCH_CONCEPTS):
        search_results[concept] = {}
        proxy_styles, proxy_objects = _build_proxy_for_concept(concept, TARGET_TYPE)
        for beta_dict in SEARCH_BETAS:
            combo_key = (f"pool{beta_dict['pooled']}_"
                         f"clip{beta_dict['clip']}_t5{beta_dict['t5']}")
            ua_t = ua_c = ira_t = ira_c = cra_t = cra_c = 0
            for style in proxy_styles:
                for obj in proxy_objects:
                    for seed in PROXY_SEEDS:
                        fname = f"{concept}__{combo_key}__{style}_{obj}_seed{seed}.jpg"
                        fp = os.path.join(PROXY_DIR, fname)
                        if not os.path.exists(fp):
                            continue
                        img = Image.open(fp).convert("RGB")
                        if TARGET_TYPE == "style":
                            ps = evaluator.classify_image(img, "style")
                            po = evaluator.classify_image(img, "object")
                            if style == concept:
                                ua_t += 1; ua_c += int(ps != concept)
                            else:
                                ira_t += 1; ira_c += int(ps == style)
                                cra_t += 1; cra_c += int(po == obj)
                        else:
                            po = evaluator.classify_image(img, "object")
                            ps = evaluator.classify_image(img, "style")
                            if obj == concept:
                                ua_t += 1; ua_c += int(po != concept)
                            else:
                                ira_t += 1; ira_c += int(po == obj)
                                cra_t += 1; cra_c += int(ps == style)
            UA = 100 * ua_c / max(ua_t, 1)
            IRA = 100 * ira_c / max(ira_t, 1)
            CRA = 100 * cra_c / max(cra_t, 1)
            score = _composite_score(UA, IRA, CRA)
            search_results[concept][combo_key] = {
                "beta": beta_dict, "UA": UA, "IRA": IRA, "CRA": CRA, "score": score}

        best_combo_key = max(search_results[concept],
                             key=lambda k: search_results[concept][k]["score"])
        best = search_results[concept][best_combo_key]
        first_key = (f"pool{SEARCH_BETAS[0]['pooled']}_"
                     f"clip{SEARCH_BETAS[0]['clip']}_t5{SEARCH_BETAS[0]['t5']}")
        last_key  = (f"pool{SEARCH_BETAS[-1]['pooled']}_"
                     f"clip{SEARCH_BETAS[-1]['clip']}_t5{SEARCH_BETAS[-1]['t5']}")
        boundary = ""
        if best_combo_key == first_key:
            boundary = " ⚠ LOW-BOUNDARY"
        elif best_combo_key == last_key:
            boundary = " ⚠ HIGH-BOUNDARY"

        best_params[concept] = {
            "beta": best["beta"],
            "clip_negative": SEARCH_CLIP_NEG,
            "step_range": list(SEARCH_STEP_RANGE),
            "clip_cap": SEARCH_CLIP_CAP,
            "proxy_UA": best["UA"], "proxy_IRA": best["IRA"],
            "proxy_CRA": best["CRA"], "proxy_score": best["score"],
            "best_combo": best_combo_key,
            "boundary_warning": bool(boundary),
        }
        print(f"  [{c_idx + 1}/{len(SEARCH_CONCEPTS)}] {concept:15s} -> "
              f"{best_combo_key:30s} UA={best['UA']:5.1f} IRA={best['IRA']:5.1f} "
              f"CRA={best['CRA']:5.1f} score={best['score']:5.1f}{boundary}")

    evaluator.llava.unload()
    steerer.pipe.to(steerer.device)
    gc.collect()
    torch.cuda.empty_cache()

    with open(best_params_path, "w") as f:
        _json.dump(best_params, f, indent=2)
    with open(search_log_path, "w") as f:
        _json.dump(search_results, f, indent=2)
    print(f"\nSaved best_params -> {best_params_path}")
    print(f"Saved search log  -> {search_log_path}")

    flagged = [c for c, p in best_params.items() if p.get("boundary_warning")]
    if flagged:
        print(f"\n⚠ {len(flagged)} concept(s) hit a grid boundary: {flagged}")
        print("  Consider extending SEARCH_BETAS in the indicated direction.")
    else:
        print("\n✓ No boundary picks; grid coverage adequate.")

# ============================================================================
# CELL 13: RUN FULL BENCHMARK -- PAPER-STYLE TABLE
# ============================================================================

if not RUN_FULL_BENCHMARK:
    print("Skipping full benchmark. Set RUN_FULL_BENCHMARK = True to run.")
else:
    import time as _time

    # Load best_params from cache if not in memory
    if "best_params" not in dir() or not isinstance(globals().get("best_params"), dict):
        if os.path.exists(best_params_path):
            with open(best_params_path) as _f:
                best_params = _json.load(_f)
            print(f"Loaded best_params from cache: {len(best_params)} concepts")
        else:
            best_params = {}
            print("⚠ No best_params cache; using global BETA fallback per concept.")

    if TARGET_TYPE == "style":
        CONCEPTS_TO_EVAL = STYLES
        make_pairs = make_style_prompts
        bench_label = "STYLE UNLEARNING"
    else:
        CONCEPTS_TO_EVAL = OBJECTS
        make_pairs = make_object_prompts
        bench_label = "OBJECT UNLEARNING"

    paper_csv = os.path.join(
        TABLES_DIR, f"paper_table_{TARGET_TYPE}_{STEERING_MODE}.csv")
    all_rows = []
    _bench_start = _time.time()

    print(f"\n{'#' * 70}")
    print(f"# FULL BENCHMARK: {bench_label} ({STEERING_MODE}, SD3.5)")
    print(f"# {len(CONCEPTS_TO_EVAL)} concepts x {len(STYLES)*len(OBJECTS)*len(EVAL_SEEDS)} images")
    print(f"#{'#' * 69}")

    # Shared baseline grid (concept-independent) for FID
    SHARED_BASELINE_DIR = os.path.join(BASELINE_DIR, "_shared_grid")
    os.makedirs(SHARED_BASELINE_DIR, exist_ok=True)
    expected = len(STYLES) * len(OBJECTS) * len(EVAL_SEEDS)
    have = len([f for f in os.listdir(SHARED_BASELINE_DIR) if f.endswith(".jpg")])
    if have < expected:
        print(f"\nGenerating shared baseline grid ({have}/{expected} present)...")
        steerer.pipe.to(steerer.device)
        for style in tqdm(STYLES, desc="Baselines"):
            for obj in OBJECTS:
                for seed in EVAL_SEEDS:
                    fp = os.path.join(SHARED_BASELINE_DIR,
                                      f"{style}_{obj}_seed{seed}.jpg")
                    if not os.path.exists(fp):
                        prompt = f"A {obj} image in {style.replace('_', ' ')} style."
                        steerer.generate(prompt, seed, vectors=None).save(fp)
        print(f"Shared baselines ready: {SHARED_BASELINE_DIR}")
    else:
        print(f"Shared baselines complete: {SHARED_BASELINE_DIR}")

    # Manual override knobs + CSV-based resume
    START_FROM = None
    SKIP_CONCEPTS = []
    completed_concepts = set()
    if os.path.exists(paper_csv):
        try:
            _existing = pd.read_csv(paper_csv)
            _existing = _existing[_existing["Concept"] != "AVERAGE"].copy()
            for _, row in _existing.iterrows():
                completed_concepts.add(str(row["Concept"]))
                all_rows.append(row.to_dict())
            if completed_concepts:
                print(f"Resume: {len(completed_concepts)} concept(s) already in CSV: "
                      f"{sorted(completed_concepts)}")
        except Exception as e:
            print(f"⚠ Could not parse CSV ({e}); starting fresh.")
            all_rows = []; completed_concepts = set()

    start_idx = 0
    if isinstance(START_FROM, int):
        start_idx = max(0, min(START_FROM, len(CONCEPTS_TO_EVAL)))
    elif isinstance(START_FROM, str) and START_FROM in CONCEPTS_TO_EVAL:
        start_idx = CONCEPTS_TO_EVAL.index(START_FROM)

    for c_idx, concept in enumerate(CONCEPTS_TO_EVAL):
        if c_idx < start_idx:
            continue
        if concept in SKIP_CONCEPTS:
            print(f"\n[{c_idx + 1}/{len(CONCEPTS_TO_EVAL)}] {concept}: SKIP_CONCEPTS")
            continue
        if concept in completed_concepts:
            print(f"\n[{c_idx + 1}/{len(CONCEPTS_TO_EVAL)}] {concept}: already in CSV")
            continue

        print(f"\n{'#' * 70}")
        print(f"# [{c_idx + 1}/{len(CONCEPTS_TO_EVAL)}] EVALUATING: {concept}")
        print(f"{'#' * 70}")

        steerer.pipe.to(steerer.device)
        vpath = os.path.join(
            VECTOR_DIR, f"{concept}_{STEERING_MODE}_diverse_vectors.pt")
        if os.path.exists(vpath):
            print(f"  Loading saved vectors from {vpath}")
            vectors = steerer.load_vectors(vpath)
        else:
            pairs = make_pairs(concept.replace('_', ' '), NUM_DIVERSE_PROMPTS)
            vectors = steerer.learn_vectors_diverse(
                prompt_pairs=pairs, seed=0, top_k=TOP_K_VECTORS, verbose=False)
            steerer.save_vectors(vectors, vpath)

        params = best_params.get(concept, {
            "beta": BETA, "clip_negative": True,
            "step_range": list(STEP_RANGE), "clip_cap": CLIP_CAP,
        })
        print(f"  Params: beta={params['beta']}, clip_cap={params['clip_cap']}")

        eval_out = evaluator.evaluate_unlearning(
            steerer=steerer, vectors=vectors,
            target_concept=concept, target_type=TARGET_TYPE,
            beta=params["beta"], clip_negative=params["clip_negative"],
            step_range=tuple(params["step_range"]), clip_cap=params["clip_cap"],
            eval_seeds=EVAL_SEEDS, save_images=True, generate_baselines=False)

        steered_dir = os.path.join(STEERED_DIR, f"{concept}_{steerer.mode}")
        fid_score = None
        try:
            fid_score = quality_metrics.calculate_fid(steered_dir, SHARED_BASELINE_DIR)
        except Exception as e:
            print(f"  ⚠ FID failed: {e}")

        clip_score = None
        try:
            imgs, prompts = [], []
            for style in STYLES:
                for obj in OBJECTS:
                    for seed in EVAL_SEEDS:
                        fp = os.path.join(steered_dir, f"{style}_{obj}_seed{seed}.jpg")
                        if os.path.exists(fp):
                            imgs.append(Image.open(fp).convert("RGB"))
                            prompts.append(
                                f"A {obj} image in {style.replace('_', ' ')} style.")
            if imgs:
                clip_score = quality_metrics.calculate_clip_score(imgs, prompts)
        except Exception as e:
            print(f"  ⚠ CLIP score failed: {e}")

        row = {
            "Concept": concept,
            "UA%":  eval_out["UA"]  * 100,
            "IRA%": eval_out["IRA"] * 100,
            "CRA%": eval_out["CRA"] * 100,
            "FID":  fid_score    if fid_score    is not None else float("nan"),
            "CLIP": clip_score   if clip_score   is not None else float("nan"),
        }
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(paper_csv, index=False)

        elapsed = _time.time() - _bench_start
        eta = elapsed / (c_idx + 1 - start_idx) * (len(CONCEPTS_TO_EVAL) - c_idx - 1)
        print(f"  {concept}: UA={row['UA%']:.1f} IRA={row['IRA%']:.1f} "
              f"CRA={row['CRA%']:.1f} FID={row['FID']:.2f} CLIP={row['CLIP']:.4f}")
        print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

        gc.collect()
        torch.cuda.empty_cache()

    # Final paper-style table + AVERAGE
    df_paper = pd.DataFrame(all_rows)
    avg_row = {
        "Concept": "AVERAGE",
        "UA%":  df_paper["UA%"].mean(),
        "IRA%": df_paper["IRA%"].mean(),
        "CRA%": df_paper["CRA%"].mean(),
        "FID":  df_paper["FID"].mean(skipna=True),
        "CLIP": df_paper["CLIP"].mean(skipna=True),
    }
    df_paper = pd.concat([df_paper, pd.DataFrame([avg_row])], ignore_index=True)
    df_paper.to_csv(paper_csv, index=False)

    print(f"\n{'=' * 70}")
    print(f"PAPER-STYLE TABLE (SD3.5 - {bench_label})")
    print(f"{'=' * 70}")
    print(df_paper.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    total_time = (_time.time() - _bench_start) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")
    print(f"Saved: {paper_csv}")
    print(f"{'=' * 70}")
