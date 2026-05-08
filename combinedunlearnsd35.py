# -*- coding: utf-8 -*-
"""CombinedUnlearnSD35.ipynb

Concept unlearning for Stable Diffusion 3.5 (MMDiT) via inference-time
activation steering. Architecture-aware: every text-conditioning entry
point is treated as a separate subspace with its own learned direction
and its own beta knob.

Run order:
  Cells 1-6   setup + architecture probe
  Cell 6.5    DIAGNOSTIC -- zero each subspace, see what each one does
              **Run this FIRST** before tuning any betas. Reveals which
              subspaces carry object identity vs style vs scene.
  Cell 7      experiment config (beta dict per subspace, informed by 6.5)
  Cell 8      learn vectors (one direction per subspace per concept)
  Cell 8B    quick steering test
  Cell 9      single-target full UA/IRA/CRA
  Cell 12B    per-concept hparam search
  Cell 13     full sweep with per-concept beta

Subspaces (each independently steered):
  pooled_clipL  -- 768-d, time_text_embed pooled input slice [0:768]
                   = CLIP-L pooled. Modulates AdaLN globally.
  pooled_clipG  -- 1280-d, time_text_embed pooled input slice [768:2048]
                   = CLIP-G pooled. Modulates AdaLN globally.
  ctx_clip      -- per-step direction in context_embedder OUTPUT applied
                   to first 77 tokens (the CLIP-L+G concatenated region).
                   Dim = caption_projection_dim (1536 medium / 2432 large).
  ctx_t5        -- per-step direction in context_embedder OUTPUT applied
                   to tokens 77..333 (the T5 region).
                   Dim = caption_projection_dim.

Modes (parallel to FLUX):
  pincer_v2      -- style : single time-averaged ctx direction per subspace
  pincer_perstep -- object: per-step ctx directions per subspace

CFG handling: under classifier-free guidance the transformer runs with
batch=2 (uncond at index 0, cond at index 1). All steering hooks apply
ONLY to the conditional position (index 1). Touching the uncond branch
contaminates the CFG reference frame and partially undoes the steering.
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
# GOOGLE DRIVE SETUP
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

# HF auth
try:
    from google.colab import userdata
    from huggingface_hub import login
    _hf_token = userdata.get("HF_TOKEN")
    if _hf_token:
        login(token=_hf_token)
        print("✓ HF authenticated via Colab secret HF_TOKEN")
except Exception:
    pass

# ============================================================================
# UNLEARNCANVAS BENCHMARK CONFIGURATION (TRACE-aligned)
# ============================================================================
STYLES = [
    "Van_Gogh", "Watercolor", "Cartoon", "Cubism", "Winter",
    "Pop_Art", "Ukiyoe", "Impressionism", "Byzantine", "Bricks"
]
OBJECTS = [
    "Architectures", "Bears", "Birds", "Butterfly", "Cats",
    "Dogs", "Fishes", "Flame", "Flowers", "Frogs",
    "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches",
    "Sea", "Statues", "Towers", "Trees", "Waterfalls"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {DEVICE}")

# Model
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
N_STEPS = 28
GUIDANCE_SCALE = 4.0
DTYPE = torch.bfloat16

EVAL_SEEDS = [188, 288, 588, 688, 888]
NUM_DIVERSE_PROMPTS = 50
TOP_K_VECTORS = 15
LEARNING_SEEDS = list(range(0, 5))

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
    n = min(num_prompts, len(IMAGENET_CLASSES))
    return [(f"{cls} with {concept}", f"{cls}") for cls in IMAGENET_CLASSES[:n]]

def make_style_prompts(concept, num_prompts=50):
    n = min(num_prompts, len(IMAGENET_CLASSES))
    return [(f"{cls}, {concept} style", f"{cls}") for cls in IMAGENET_CLASSES[:n]]

RUN_FULL_BENCHMARK = False

for subdir in ["steering_vectors", "results", "baseline_images", "steered_images", "tables"]:
    os.makedirs(os.path.join(ROOT_DIR, subdir), exist_ok=True)

VECTOR_DIR   = os.path.join(ROOT_DIR, "steering_vectors")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
BASELINE_DIR = os.path.join(ROOT_DIR, "baseline_images")
STEERED_DIR  = os.path.join(ROOT_DIR, "steered_images")
TABLES_DIR   = os.path.join(ROOT_DIR, "tables")

print("\n" + "=" * 70)
print("SD3.5 UNLEARNCANVAS CONFIGURATION")
print("=" * 70)
print(f"Model:           {MODEL_ID}")
print(f"Steps:           {N_STEPS}")
print(f"Guidance scale:  {GUIDANCE_SCALE}")
print(f"Eval seeds:      {EVAL_SEEDS}")
print("=" * 70)

# ============================================================================
# CELL 3: SD35STEERING CLASS (per-subspace, diagnostic-first)
# ============================================================================

class SD35Steering:
    """Architecture-aware concept steering for Stable Diffusion 3.5 (MMDiT).

    FIVE independently-controllable subspaces (one direction per concept,
    one beta knob each):

      pooled_clipL  : 768-d, time_text_embed pooled input slice [0:768]
                      (CLIP-L pooled, raw, BEFORE the modulation MLP)
      pooled_clipG  : 1280-d, slice [768:2048]
                      (CLIP-G pooled, raw, BEFORE the modulation MLP)
      tte_out       : caption_projection_dim, time_text_embed OUTPUT
                      (= the AdaLN modulation signal AFTER the SiLU MLP).
                      TRACE Appendix D.1: "Despite operating on a pooled
                      representation, we found it necessary to intervene at
                      this layer as well to ensure effective suppression".
                      This is the layer SD3.5 needs that FLUX doesn't --
                      because SD3.5's modulation MLP nonlinearly mixes the
                      input, so subtracting at the input doesn't cleanly
                      remove the concept downstream.
      ctx_clip      : context_embedder OUTPUT, first 77 tokens
                      (CLIP-L+G concatenated joint sequence after projection)
      ctx_t5        : context_embedder OUTPUT, tokens 77..333
                      (T5 sequence after projection)

    Modes (parallel to FLUX):
      pincer_v2      -- style : single time-averaged direction per subspace
      pincer_perstep -- object: per-step directions per subspace

    CFG handling: under classifier-free guidance the transformer runs with
    batch=2 (uncond at index 0, cond at index 1). All steering hooks apply
    ONLY to the conditional position (index 1). Touching the uncond branch
    contaminates the CFG reference frame and partially undoes the steering.

    Use diagnose() before tuning betas to find which subspaces matter for
    your concept. Diagnostic now includes zeroing tte_out so we can verify
    TRACE's finding on our specific setup.
    """

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

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.VALID_MODES}.")

        self.target_layers = {
            "context_embedder": pipe.transformer.context_embedder,
            "time_text_embed":  pipe.transformer.time_text_embed,
        }

        self.caption_dim = pipe.transformer.config.caption_projection_dim
        n_blocks = len(pipe.transformer.transformer_blocks)

        print(f"SD35Steering initialized (mode={mode}, CFG={guidance_scale})")
        print(f"  caption_projection_dim : {self.caption_dim}")
        print(f"  pooled split           : CLIP-L [0:768] + CLIP-G [768:2048]")
        print(f"  ctx split              : CLIP region [0:77] + T5 region [77:333]")
        print(f"  tte_out (NEW)          : modulation signal after SiLU MLP")
        print(f"                           dim = caption_projection_dim ({self.caption_dim})")
        print(f"  transformer blocks     : {n_blocks}")
        print(f"  Run diagnose() FIRST to identify which subspaces matter.")
        print(f"  TRACE App. D.1 says SD3.5 needs intervention at the modulation MLP.")
        print(f"  Our previous file missed this -- now added as 'tte_out' subspace.")

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
        tok = self.pipe.tokenizer(prompt, padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt")
        return tok.attention_mask.to(self.device)   # (1, 77)

    def _get_t5_mask(self, prompt):
        tok = self.pipe.tokenizer_3(prompt, padding="max_length", max_length=256,
                                    truncation=True, return_tensors="pt")
        return tok.attention_mask.to(self.device)   # (1, 256)

    def _masked_mean(self, act, mask):
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
    # DIAGNOSE: zero each subspace and see what survives
    # ==================================================================
    @torch.no_grad()
    def diagnose(self, prompt, seed):
        """
        Generate baseline + 8 zeroing variants for a single (prompt, seed).
        Each variant zeroes a specific subspace. Comparing the resulting
        images reveals which subspace carries which kind of information.

        Returns: dict {label: PIL.Image}.

        Interpretation:
          baseline             : reference
          zero_pooled_all      : zeros pooled INPUT to time_text_embed (pre-MLP)
          zero_pooled_clipL    : zeros only CLIP-L pooled slice [0:768]
          zero_pooled_clipG    : zeros only CLIP-G pooled slice [768:2048]
          zero_tte_out  (NEW)  : zeros the OUTPUT of time_text_embed (post-MLP)
                                 -- the actual AdaLN modulation signal. Per
                                 TRACE App. D.1, this is the layer that
                                 needs intervention on SD3.5.
          zero_seq_clip_region : zeros first 77 tokens of context_embedder INPUT
          zero_seq_t5          : zeros tokens 77..333 of context_embedder INPUT
          zero_ctx_out_clip    : zeros first 77 tokens of context_embedder OUTPUT
          zero_ctx_out_t5      : zeros tokens 77..333 of context_embedder OUTPUT

        If zero_tte_out destroys the dog while zero_pooled_all does not, that
        confirms TRACE's finding empirically -- the modulation MLP's output
        carries concept signal that the input perturbation alone misses.
        """
        def make_pre_pooled(slice_a, slice_b):
            def hook(module, args):
                timestep, pooled = args
                cond_idx = 1 if pooled.shape[0] >= 2 else 0
                pooled = pooled.clone()
                pooled[cond_idx, slice_a:slice_b] = 0
                return (timestep, pooled)
            return hook

        def make_pre_ctx_input(token_a, token_b, dim_a=None, dim_b=None):
            def hook(module, args):
                x = args[0].clone()   # (B, 333, 4096)
                cond_idx = 1 if x.shape[0] >= 2 else 0
                if dim_a is None:
                    x[cond_idx, token_a:token_b, :] = 0
                else:
                    x[cond_idx, token_a:token_b, dim_a:dim_b] = 0
                return (x,) + args[1:]
            return hook

        def make_post_ctx_output(token_a, token_b):
            def hook(module, inputs, output):
                out = output.clone()   # (B, 333, caption_dim)
                cond_idx = 1 if out.shape[0] >= 2 else 0
                out[cond_idx, token_a:token_b, :] = 0
                return out
            return hook

        def make_post_tte_output():
            """Zero time_text_embed OUTPUT (= AdaLN modulation signal).
            This is the post-MLP, post-SiLU signal -- TRACE's finding."""
            def hook(module, inputs, output):
                out = output.clone()   # (B, caption_dim)
                cond_idx = 1 if out.shape[0] >= 2 else 0
                out[cond_idx, :] = 0
                return out
            return hook

        # Each experiment is a list of hooks to register.
        experiments = [
            ("baseline",            []),
            ("zero_pooled_all",     [("time_text_embed", "pre",
                                      make_pre_pooled(0, 2048))]),
            ("zero_pooled_clipL",   [("time_text_embed", "pre",
                                      make_pre_pooled(0, 768))]),
            ("zero_pooled_clipG",   [("time_text_embed", "pre",
                                      make_pre_pooled(768, 2048))]),
            # NEW: zero the modulation MLP's OUTPUT (TRACE finding)
            ("zero_tte_out",        [("time_text_embed", "post",
                                      make_post_tte_output())]),
            ("zero_seq_clip_region", [("context_embedder", "pre",
                                      make_pre_ctx_input(0, 77))]),
            ("zero_seq_t5",         [("context_embedder", "pre",
                                      make_pre_ctx_input(77, 333))]),
            ("zero_ctx_out_clip",   [("context_embedder", "post",
                                      make_post_ctx_output(0, 77))]),
            ("zero_ctx_out_t5",     [("context_embedder", "post",
                                      make_post_ctx_output(77, 333))]),
        ]

        results = {}
        for label, hook_specs in tqdm(experiments, desc="Diagnostic"):
            self._clear_hooks()
            for layer_name, kind, hook_fn in hook_specs:
                module = self.target_layers[layer_name]
                if kind == "pre":
                    self._handles.append(module.register_forward_pre_hook(hook_fn))
                else:
                    self._handles.append(module.register_forward_hook(hook_fn))
            try:
                img = self._run_pipe_base(prompt, seed)
            finally:
                self._clear_hooks()
            results[label] = img
        return results

    # ==================================================================
    # LEARN VECTORS: one direction per subspace
    # ==================================================================
    @torch.no_grad()
    def learn_vectors_diverse(self, prompt_pairs, seed=0, top_k=15, verbose=True):
        """
        Capture per-subspace directions over diverse pairs (CASteer Appendix C).

        For each (pos, neg) pair:
          * Run the pipeline once for pos and once for neg, same seed.
          * Hook time_text_embed to record the 2048-d pooled embedding
            (conditional position only).
          * Hook context_embedder OUTPUT to record (333, caption_dim)
            at every denoising step (conditional position only).
        After all N pairs:
          * pooled_clipL  : mean(pos[:768] - neg[:768]) over pairs, normed.
          * pooled_clipG  : mean(pos[768:] - neg[768:]) over pairs, normed.
          * ctx_clip      : per-step masked-mean over first 77 tokens.
            pincer_v2  : single direction averaged over steps.
            pincer_perstep : direction per step.
          * ctx_t5        : per-step masked-mean over T5 region (77..333).

        Returns dict with keys: pooled_clipL, pooled_clipG, ctx_clip, ctx_t5.
        Each value is {step: tensor}; for pooled, only step 0 is populated.
        """
        n_pairs = len(prompt_pairs)
        if verbose:
            print(f"Learning from {n_pairs} diverse pairs (seed={seed}, mode={self.mode})")

        ctx_clip_acc = {step: None for step in range(self.n_steps)}
        ctx_t5_acc   = {step: None for step in range(self.n_steps)}
        # NEW: per-step accumulator for time_text_embed OUTPUT (modulation signal)
        tte_out_acc  = {step: None for step in range(self.n_steps)}
        pooled_acc = None

        cap_ctx = {}    # {step: (333, caption_dim)}
        cap_pooled = {}
        cap_tte = {}    # {step: (caption_dim,)} -- NEW

        def _ctx_hook(module, inp, out):
            step = self._current_step + 1
            if 0 <= step < self.n_steps:
                cond_idx = 1 if out.shape[0] >= 2 else 0
                cap_ctx[step] = out[cond_idx].detach().float().cpu()

        def _pooled_pre_hook(module, args):
            if "pooled" not in cap_pooled:
                pooled = args[1]
                cond_idx = 1 if pooled.shape[0] >= 2 else 0
                cap_pooled["pooled"] = pooled[cond_idx].detach().float().cpu()

        def _tte_post_hook(module, inp, out):
            """Capture time_text_embed OUTPUT (the post-MLP modulation signal).
            Per TRACE App. D.1, this is the layer SD3.5 needs intervention on."""
            step = self._current_step + 1
            if 0 <= step < self.n_steps:
                cond_idx = 1 if out.shape[0] >= 2 else 0
                cap_tte[step] = out[cond_idx].detach().float().cpu()

        def _capture(prompt, seed_):
            cap_ctx.clear()
            cap_pooled.clear()
            cap_tte.clear()
            self._clear_hooks()
            self._handles.append(
                self.target_layers["context_embedder"].register_forward_hook(_ctx_hook))
            self._handles.append(
                self.target_layers["time_text_embed"].register_forward_pre_hook(_pooled_pre_hook))
            self._handles.append(
                self.target_layers["time_text_embed"].register_forward_hook(_tte_post_hook))
            try:
                self._run_pipe_base(prompt, seed_)
            finally:
                self._clear_hooks()
            return dict(cap_ctx), cap_pooled.get("pooled"), dict(cap_tte)

        for pair_idx, (pos_p, neg_p) in enumerate(
            tqdm(prompt_pairs, desc="Diverse pairs", disable=not verbose)
        ):
            pos_clip_mask = self._get_clip_l_mask(pos_p)
            pos_t5_mask   = self._get_t5_mask(pos_p)
            neg_clip_mask = self._get_clip_l_mask(neg_p)
            neg_t5_mask   = self._get_t5_mask(neg_p)

            pos_ctx, pos_pooled, pos_tte = _capture(pos_p, seed)
            neg_ctx, neg_pooled, neg_tte = _capture(neg_p, seed)

            for step in range(self.n_steps):
                if step in pos_ctx and step in neg_ctx:
                    pos_seq = pos_ctx[step]
                    neg_seq = neg_ctx[step]
                    seq_len = min(pos_seq.shape[0], neg_seq.shape[0])
                    pos_seq = pos_seq[:seq_len]
                    neg_seq = neg_seq[:seq_len]

                    # CLIP region (first 77 tokens), masked mean
                    pos_clip_pool = self._masked_mean(
                        pos_seq[:77].unsqueeze(0), pos_clip_mask)
                    neg_clip_pool = self._masked_mean(
                        neg_seq[:77].unsqueeze(0), neg_clip_mask)
                    d_clip = pos_clip_pool - neg_clip_pool
                    ctx_clip_acc[step] = d_clip if ctx_clip_acc[step] is None else ctx_clip_acc[step] + d_clip

                    # T5 region (tokens 77..333), masked mean
                    pos_t5_pool = self._masked_mean(
                        pos_seq[77:77 + 256].unsqueeze(0), pos_t5_mask)
                    neg_t5_pool = self._masked_mean(
                        neg_seq[77:77 + 256].unsqueeze(0), neg_t5_mask)
                    d_t5 = pos_t5_pool - neg_t5_pool
                    ctx_t5_acc[step] = d_t5 if ctx_t5_acc[step] is None else ctx_t5_acc[step] + d_t5

                # tte_out diff per step (NEW)
                if step in pos_tte and step in neg_tte:
                    d_tte = pos_tte[step] - neg_tte[step]
                    tte_out_acc[step] = d_tte if tte_out_acc[step] is None else tte_out_acc[step] + d_tte

            if pos_pooled is not None and neg_pooled is not None:
                d_pool = pos_pooled - neg_pooled
                pooled_acc = d_pool if pooled_acc is None else pooled_acc + d_pool

            if verbose and (pair_idx + 1) % 10 == 0:
                print(f"  Completed {pair_idx + 1}/{n_pairs} pairs")

        vectors = {}

        # Pooled split into CLIP-L and CLIP-G
        if pooled_acc is not None:
            avg = pooled_acc / n_pairs
            clipL = avg[:768]
            clipG = avg[768:]
            vectors["pooled_clipL"] = {0: (clipL / (clipL.norm() + 1e-8)).to(self.device, dtype=DTYPE)}
            vectors["pooled_clipG"] = {0: (clipG / (clipG.norm() + 1e-8)).to(self.device, dtype=DTYPE)}

        # ctx directions + tte_out directions (per-step or single, by mode)
        if self.mode == "pincer_v2":
            valid_clip = [v for v in ctx_clip_acc.values() if v is not None]
            valid_t5   = [v for v in ctx_t5_acc.values() if v is not None]
            valid_tte  = [v for v in tte_out_acc.values() if v is not None]
            if valid_clip:
                avg = sum(valid_clip) / (len(valid_clip) * n_pairs)
                vectors["ctx_clip"] = {0: (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)}
            if valid_t5:
                avg = sum(valid_t5) / (len(valid_t5) * n_pairs)
                vectors["ctx_t5"]   = {0: (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)}
            if valid_tte:
                avg = sum(valid_tte) / (len(valid_tte) * n_pairs)
                vectors["tte_out"]  = {0: (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)}
        else:
            ctx_clip_dirs, ctx_t5_dirs, tte_out_dirs = {}, {}, {}
            for step in range(self.n_steps):
                if ctx_clip_acc[step] is not None:
                    avg = ctx_clip_acc[step] / n_pairs
                    ctx_clip_dirs[step] = (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)
                if ctx_t5_acc[step] is not None:
                    avg = ctx_t5_acc[step] / n_pairs
                    ctx_t5_dirs[step] = (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)
                if tte_out_acc[step] is not None:
                    avg = tte_out_acc[step] / n_pairs
                    tte_out_dirs[step] = (avg / (avg.norm() + 1e-8)).to(self.device, dtype=DTYPE)
            vectors["ctx_clip"] = ctx_clip_dirs
            vectors["ctx_t5"]   = ctx_t5_dirs
            vectors["tte_out"]  = tte_out_dirs

        if verbose:
            print(f"\n{'='*70}")
            print(f"SD3.5 Steering Vectors ({self.mode}, {n_pairs} pairs)")
            print(f"{'='*70}")
            for k, step_vecs in vectors.items():
                sample = next(iter(step_vecs.values()))
                print(f"  {k:<18} #steps={len(step_vecs):<3} dim={sample.shape[-1]:<5} sample_norm={float(sample.norm()):.4f}")
            print(f"{'='*70}\n")

        return vectors

    # ==================================================================
    # APPLY VECTORS: per-subspace hooks
    # ==================================================================
    @contextmanager
    def apply_vectors(self, vectors, beta=2.0, clip_negative=True,
                      step_range=None, clip_cap=None):
        """Apply steering. beta can be a single float (applied to all subspaces)
        or a dict with any subset of:
          {pooled_clipL, pooled_clipG, tte_out, ctx_clip, ctx_t5}

        tte_out (NEW) steers the OUTPUT of time_text_embed -- the AdaLN
        modulation signal after the SiLU MLP. Per TRACE App. D.1, this is
        the layer SD3.5 needs intervention on; subtracting at the input
        only doesn't propagate cleanly through the MLP nonlinearity.

        Subspaces with beta=0 (or missing from vectors) are not steered.
        Steering applies ONLY to the conditional batch position under CFG.
        """
        if isinstance(beta, dict):
            b_clipL = beta.get("pooled_clipL", 0.0)
            b_clipG = beta.get("pooled_clipG", 0.0)
            b_tte   = beta.get("tte_out", 0.0)
            b_ctx_clip = beta.get("ctx_clip", 0.0)
            b_ctx_t5   = beta.get("ctx_t5", 0.0)
        else:
            b_clipL = b_clipG = b_tte = b_ctx_clip = b_ctx_t5 = float(beta)

        if step_range is None:
            def _in_range(step): return True
        else:
            s_start, s_end = step_range
            def _in_range(step): return s_start <= step < s_end

        def make_pooled_hook(direction_L, b_L, direction_G, b_G):
            """Pre-hook on time_text_embed: subtracts CLIP-L slice and CLIP-G slice
            independently from the pooled vector."""
            def hook(module, args):
                step = self._current_step + 1
                if not _in_range(step):
                    return None
                timestep, pooled = args
                cond_idx = 1 if pooled.shape[0] >= 2 else 0
                pooled_new = pooled.clone()
                cond = pooled_new[cond_idx]

                if direction_L is not None and b_L > 0:
                    d = direction_L.to(cond.device, cond.dtype)
                    score = (cond[:768] @ d)
                    if clip_negative:
                        score = score.clamp(min=0.0)
                    if clip_cap is None:
                        eff = float(b_L) * score
                    else:
                        eff = min(float(b_L), float(clip_cap)) * score
                    cond = cond.clone()
                    cond[:768] = cond[:768] - eff * d

                if direction_G is not None and b_G > 0:
                    d = direction_G.to(cond.device, cond.dtype)
                    score = (cond[768:] @ d)
                    if clip_negative:
                        score = score.clamp(min=0.0)
                    if clip_cap is None:
                        eff = float(b_G) * score
                    else:
                        eff = min(float(b_G), float(clip_cap)) * score
                    cond = cond.clone()
                    cond[768:] = cond[768:] - eff * d

                pooled_new[cond_idx] = cond
                return (timestep, pooled_new)
            return hook

        def make_ctx_hook(layer_clip_vecs, layer_t5_vecs, b_clip, b_t5):
            """Output hook on context_embedder. Modifies CLIP and T5 regions
            of the conditional position independently."""
            def hook(module, inputs, output):
                step = self._current_step + 1
                if not _in_range(step):
                    return output
                cond_idx = 1 if output.shape[0] >= 2 else 0
                out = output.clone()

                if layer_clip_vecs and b_clip > 0:
                    d_clip = layer_clip_vecs.get(step)
                    if d_clip is None:
                        d_clip = layer_clip_vecs.get(0)
                    if d_clip is not None:
                        d = d_clip.to(out.device, out.dtype)
                        cc = out[cond_idx, :77]
                        score = cc @ d
                        if clip_negative:
                            score = score.clamp(min=0.0)
                        update = (b_clip * score).unsqueeze(-1) * d
                        out[cond_idx, :77] = cc - update

                if layer_t5_vecs and b_t5 > 0:
                    d_t5 = layer_t5_vecs.get(step)
                    if d_t5 is None:
                        d_t5 = layer_t5_vecs.get(0)
                    if d_t5 is not None:
                        d = d_t5.to(out.device, out.dtype)
                        ct = out[cond_idx, 77:333]
                        score = ct @ d
                        if clip_negative:
                            score = score.clamp(min=0.0)
                        update = (b_t5 * score).unsqueeze(-1) * d
                        out[cond_idx, 77:333] = ct - update

                return out
            return hook

        def make_tte_post_hook(layer_vecs, b):
            """Post-hook on time_text_embed: subtract the dog direction from
            the post-MLP modulation signal of the conditional position.
            This is the TRACE-aligned intervention that SD3.5 needs."""
            def hook(module, inputs, output):
                step = self._current_step + 1
                if not _in_range(step):
                    return output
                cond_idx = 1 if output.shape[0] >= 2 else 0
                d_tte = layer_vecs.get(step)
                if d_tte is None:
                    d_tte = layer_vecs.get(0)
                if d_tte is None:
                    return output
                out = output.clone()
                d = d_tte.to(out.device, out.dtype)
                cond = out[cond_idx]
                score = cond @ d
                if clip_negative:
                    score = score.clamp(min=0.0)
                if clip_cap is None:
                    eff = float(b) * score
                else:
                    eff = min(float(b), float(clip_cap)) * score
                out[cond_idx] = cond - eff * d
                return out
            return hook

        try:
            self._clear_hooks()

            # Pre-hook on time_text_embed input: pooled CLIP-L / CLIP-G
            pooled_L = vectors.get("pooled_clipL", {}).get(0)
            pooled_G = vectors.get("pooled_clipG", {}).get(0)
            if (pooled_L is not None and b_clipL > 0) or (pooled_G is not None and b_clipG > 0):
                self._handles.append(
                    self.target_layers["time_text_embed"].register_forward_pre_hook(
                        make_pooled_hook(pooled_L, b_clipL, pooled_G, b_clipG)))

            # Post-hook on time_text_embed OUTPUT: tte_out (TRACE finding).
            # This is the AdaLN modulation signal AFTER the SiLU MLP.
            tte_vecs = vectors.get("tte_out", {})
            if tte_vecs and b_tte > 0:
                self._handles.append(
                    self.target_layers["time_text_embed"].register_forward_hook(
                        make_tte_post_hook(tte_vecs, b_tte)))

            # Output hook on context_embedder: CLIP / T5 regions.
            ctx_clip_vecs = vectors.get("ctx_clip", {})
            ctx_t5_vecs   = vectors.get("ctx_t5", {})
            if (ctx_clip_vecs and b_ctx_clip > 0) or (ctx_t5_vecs and b_ctx_t5 > 0):
                self._handles.append(
                    self.target_layers["context_embedder"].register_forward_hook(
                        make_ctx_hook(ctx_clip_vecs, ctx_t5_vecs, b_ctx_clip, b_ctx_t5)))
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
        print(f"Saved vectors -> {filepath}")

    def load_vectors(self, filepath):
        save = torch.load(filepath, map_location=self.device)
        return {k: {step: t.to(self.device, dtype=DTYPE) for step, t in v.items()}
                for k, v in save.items()}


print("✓ SD35Steering class defined!")

# ============================================================================
# CELL 4: QUALITY METRICS
# ============================================================================

class QualityMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading CLIP for quality metrics...")
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
                im = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                t = clip.tokenize([prompt], truncate=True).to(self.device)
                imf = self.clip_model.encode_image(im)
                tf = self.clip_model.encode_text(t)
                imf = imf / imf.norm(dim=-1, keepdim=True)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                scores.append((imf @ tf.T).item())
        return float(np.mean(scores))

    def calculate_fid(self, real_path, generated_path):
        try:
            return fid.compute_fid(real_path, generated_path, mode="clean",
                                   num_workers=0, batch_size=8,
                                   device=torch.device(self.device))
        except Exception as e:
            print(f"⚠ FID error: {e}")
            return None


print("✓ QualityMetrics defined!")

# ============================================================================
# CELL 4B: LLAVA CLASSIFIER (TRACE format)
# ============================================================================

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class LLaVAClassifier:
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
        conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        formatted = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self.processor(images=image, text=formatted,
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
        if debug: print(f"LLaVA style: '{resp}'")
        idx = self._parse_number(resp, len(styles))
        if idx is not None: return styles[idx]
        rl = resp.lower()
        for s in styles:
            if s.lower().replace('_', ' ') in rl: return s
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
        if debug: print(f"LLaVA object: '{resp}'")
        idx = self._parse_number(resp, len(objects))
        if idx is not None: return objects[idx]
        rl = resp.lower()
        for o in objects:
            if o.lower().replace('_', ' ') in rl: return o
        return None


print("✓ LLaVAClassifier defined!")

# ============================================================================
# CELL 5: UNLEARNCANVAS EVALUATOR (TRACE-correct CRA scoring)
# ============================================================================

class UnlearnCanvasEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.llava = LLaVAClassifier(device=device)
        print("✓ Evaluator ready (LLaVA-1.6-Vicuna-7B)")

    def classify_image(self, image, domain="style"):
        return (self.llava.classify_style(image) if domain == "style"
                else self.llava.classify_object(image))

    def evaluate_unlearning(
        self, steerer, vectors, target_concept, target_type="object",
        beta=2.0, clip_negative=True, step_range=None, clip_cap=None,
        eval_seeds=None, save_images=True, output_dir=None,
        generate_baselines=False
    ):
        eval_seeds = eval_seeds or EVAL_SEEDS
        output_dir = output_dir or os.path.join(STEERED_DIR, f"{target_concept}_{steerer.mode}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {target_concept} ({target_type})")
        print(f"{'=' * 70}")

        test_cases = []
        for style in STYLES:
            for obj in OBJECTS:
                for seed in eval_seeds:
                    fname = f"{style}_{obj}_seed{seed}.jpg"
                    prompt = f"A {obj} image in {style.replace('_', ' ')} style."
                    test_cases.append({
                        "prompt": prompt, "seed": seed,
                        "gt_style": style, "gt_object": obj, "filename": fname})
        total = len(test_cases)

        skipped = generated = 0
        print(f"\n--- PHASE 1: GENERATION ({total} images) ---")
        print(f"Steering: beta={beta}, step_range={step_range}, clip_cap={clip_cap}")
        for case in tqdm(test_cases, desc="Generating"):
            sp = os.path.join(output_dir, case["filename"])
            if os.path.exists(sp):
                skipped += 1; continue
            img = steerer.generate(case["prompt"], case["seed"], vectors=vectors,
                                   beta=beta, clip_negative=clip_negative,
                                   step_range=step_range, clip_cap=clip_cap)
            img.save(sp); generated += 1
        print(f"Phase 1: {generated} generated, {skipped} skipped")

        print("\nFreeing SD3.5 VRAM...")
        steerer.pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n--- PHASE 2: CLASSIFICATION ---")
        results = {"target_correct": 0, "target_total": 0,
                   "ira_correct": 0, "ira_total": 0,
                   "cra_correct": 0, "cra_total": 0, "prompts": []}
        for i, case in enumerate(tqdm(test_cases, desc="Classifying")):
            ip = os.path.join(output_dir, case["filename"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            results["prompts"].append(case["prompt"])
            gs, go = case["gt_style"], case["gt_object"]
            if target_type == "style":
                ps = self.classify_image(img, "style")
                po = self.classify_image(img, "object")
                if gs == target_concept:
                    results["target_total"] += 1
                    if ps == target_concept: results["target_correct"] += 1
                else:
                    results["ira_total"] += 1
                    if ps == gs: results["ira_correct"] += 1
                    results["cra_total"] += 1
                    if po == go: results["cra_correct"] += 1
            else:
                po = self.classify_image(img, "object")
                ps = self.classify_image(img, "style")
                if go == target_concept:
                    results["target_total"] += 1
                    if po == target_concept: results["target_correct"] += 1
                else:
                    results["ira_total"] += 1
                    if po == go: results["ira_correct"] += 1
                    results["cra_total"] += 1
                    if ps == gs: results["cra_correct"] += 1

            if (i + 1) % 50 == 0:
                _ua = 1.0 - (results["target_correct"] / max(results["target_total"], 1))
                _ira = results["ira_correct"] / max(results["ira_total"], 1)
                _cra = results["cra_correct"] / max(results["cra_total"], 1)
                print(f"  [{i+1}/{total}] UA={_ua:.1%} IRA={_ira:.1%} CRA={_cra:.1%}")

        print("\nUnloading LLaVA, reloading SD3.5...")
        self.llava.unload()
        steerer.pipe.to(steerer.device)

        ua = 1.0 - (results["target_correct"] / max(results["target_total"], 1))
        ira = results["ira_correct"] / max(results["ira_total"], 1)
        cra = results["cra_correct"] / max(results["cra_total"], 1)

        print(f"\n{target_concept}: UA={ua:.2%} IRA={ira:.2%} CRA={cra:.2%}")
        return {"UA": ua, "IRA": ira, "CRA": cra,
                "target_concept": target_concept, "target_type": target_type,
                "beta": beta, "n_images": total, "prompts": results["prompts"]}


print("✓ UnlearnCanvasEvaluator defined!")

# ============================================================================
# CELL 6: LOAD MODELS + ARCHITECTURE PROBE
# ============================================================================

print("\nLoading SD3.5 pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
print(f"✓ Pipeline loaded: {MODEL_ID}")

# Architecture probe
print("\nArchitecture probe:")
_t = pipe.transformer
_caption_dim = _t.config.caption_projection_dim
_pooled_dim = _t.config.pooled_projection_dim
_n_layers = len(_t.transformer_blocks)
print(f"  caption_projection_dim : {_caption_dim}    (1536 medium / 2432 large)")
print(f"  pooled_projection_dim  : {_pooled_dim}    (expected 2048)")
print(f"  num_transformer_blocks : {_n_layers}      (24 medium / 38 large)")
assert _pooled_dim == 2048, f"pooled_projection_dim={_pooled_dim} != 2048"
assert isinstance(_t.context_embedder, torch.nn.Linear)
assert _t.context_embedder.in_features == 4096
assert _t.context_embedder.out_features == _caption_dim
print(f"  ✓ All assumptions hold")

# ===== EXPERIMENT TARGET =====
TARGET_CONCEPT = "Dogs"
TARGET_TYPE    = "object"

if TARGET_TYPE == "style":
    STEERING_MODE = "pincer_v2"
else:
    STEERING_MODE = "pincer_perstep"

print(f"\nInitializing SD35Steering (mode={STEERING_MODE}, CFG={GUIDANCE_SCALE})")
steerer = SD35Steering(pipe, device=DEVICE, n_steps=N_STEPS,
                      mode=STEERING_MODE, guidance_scale=GUIDANCE_SCALE)

print("\nInitializing evaluators...")
evaluator = UnlearnCanvasEvaluator(device=DEVICE)
quality_metrics = QualityMetrics(device=DEVICE)
print("\n✓ ALL MODELS LOADED")

# ============================================================================
# CELL 6.5: SUBSPACE DIAGNOSTIC -- RUN THIS FIRST
# ============================================================================
"""
Generate baseline + 7 zeroing variants of a single test prompt to identify
which subspaces carry which kinds of information in SD3.5.

Interpretation guide (look at the panel and judge):
  zero_pooled_all      : if image is destroyed -> pooled-modulation is the
                         dominant text-conditioning pathway
  zero_pooled_clipL    : changes -> CLIP-L pooled contributes to <X>
  zero_pooled_clipG    : changes -> CLIP-G pooled contributes to <X>
                         compare clipL vs clipG to see which dominates for
                         your target concept type
  zero_seq_clip_region : changes -> CLIP sequence (in context) carries <X>
  zero_seq_t5          : changes -> T5 sequence carries <X>
  zero_ctx_out_clip    : same as zero_seq_clip_region but at OUTPUT (after
                         context_embedder Linear projection)
  zero_ctx_out_t5      : same as zero_seq_t5 but at OUTPUT

Read the panel like:
  * Pick a prompt with a clear OBJECT (e.g. "A dog in Van Gogh style").
  * If zero_pooled_clipL kills the dog -> CLIP-L pooled carries object
    identity. Steer pooled_clipL with high beta for object unlearning.
  * If zero_seq_t5 kills the Van Gogh style -> T5 carries style. Steer
    ctx_t5 with high beta for style unlearning.
  * If multiple subspaces all change the image, the concept is distributed
    -- you'll need to steer multiple subspaces jointly.
"""

DIAG_TEST_PROMPT = "A photo of a dog in Van Gogh style"
DIAG_TEST_SEED = 42

print(f"\nRunning subspace diagnostic on '{DIAG_TEST_PROMPT}', seed={DIAG_TEST_SEED}...")
print("This generates 8 images (~3-5 min on A100). Each variant zeroes a different subspace.")
diag_results = steerer.diagnose(DIAG_TEST_PROMPT, DIAG_TEST_SEED)

# Display panel
n = len(diag_results)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.flatten() if nrows > 1 else (axes if hasattr(axes, '__len__') else [axes])
baseline_arr = np.array(diag_results["baseline"]).astype(float)
for i, (label, img) in enumerate(diag_results.items()):
    ax = axes[i]
    ax.imshow(img)
    if label == "baseline":
        ax.set_title(label, fontsize=11, fontweight="bold")
    else:
        diff = np.abs(np.array(img).astype(float) - baseline_arr)
        pct = (diff > 5.0).mean() * 100
        color = "darkred" if pct > 50 else "orange" if pct > 20 else "gray"
        ax.set_title(f"{label}\n{pct:.0f}% pixels changed",
                     fontsize=10, color=color)
    ax.axis("off")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.suptitle(f"SD3.5 Subspace Diagnostic\nPrompt: '{DIAG_TEST_PROMPT}'",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("INTERPRETATION CHECKLIST")
print("=" * 70)
print("Compare each variant against the baseline:")
print("  * Which variant(s) DESTROY the dog (object)?")
print("    -> those subspaces carry OBJECT identity. Steer them for object unlearning.")
print("  * Which variant(s) DESTROY the Van Gogh style?")
print("    -> those subspaces carry STYLE. Steer them for style unlearning.")
print("  * Which variant(s) leave the image essentially unchanged?")
print("    -> those subspaces are NOT carrying the concept. Don't waste steering on them.")
print("=" * 70)
print("\nAdjust BETA in Cell 7 based on what you see above.")

# ============================================================================
# CELL 7: EXPERIMENT CONFIGURATION (PER-SUBSPACE BETA)
# ============================================================================
"""
Beta per subspace. Set high beta on subspaces the diagnostic showed
matter; leave others at 0.

Defaults below are starting points. ADJUST based on Cell 6.5 findings:
  * If pooled_clipL had the biggest effect on the dog (object), bump
    its beta higher and consider lowering pooled_clipG.
  * If ctx_t5 had the biggest effect on style, bump it higher for
    style unlearning.
"""
if TARGET_TYPE == "style":
    # Style: previous diagnostic showed ctx_clip is the primary lever.
    # Adding tte_out as a secondary lever per TRACE finding -- the
    # modulation signal also encodes some style information.
    BETA = {
        "pooled_clipL": 0.0,
        "pooled_clipG": 0.0,
        "tte_out":      4.0,    # NEW: TRACE-aligned modulation MLP intervention
        "ctx_clip":     8.0,    # primary lever from diagnostic
        "ctx_t5":       0.0,    # diagnostic showed T5 doesn't carry style here
    }
    STEP_RANGE = (0, N_STEPS)
    CLIP_CAP = 1.0
else:
    # Object: identity is redundantly distributed. Joint multi-subspace.
    # tte_out (NEW) is the missing piece that lets steering propagate
    # through the modulation MLP cleanly -- TRACE App. D.1 says this is
    # the layer SD3.5 needs intervention on. Initial high beta for it.
    BETA = {
        "pooled_clipL": 5.0,
        "pooled_clipG": 10.0,
        "tte_out":     12.0,    # NEW: post-MLP modulation signal -- the missing lever
        "ctx_clip":    10.0,
        "ctx_t5":       2.0,
    }
    STEP_RANGE = (0, N_STEPS)
    CLIP_CAP = None

OUTPUT_DIR = os.path.join(RESULTS_DIR, f"{TARGET_CONCEPT}_{STEERING_MODE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if TARGET_TYPE == "style":
    DIVERSE_PROMPT_PAIRS = make_style_prompts(
        TARGET_CONCEPT.replace('_', ' '), NUM_DIVERSE_PROMPTS)
else:
    DIVERSE_PROMPT_PAIRS = make_object_prompts(
        TARGET_CONCEPT.replace('_', ' '), NUM_DIVERSE_PROMPTS)

print("=" * 70)
print(f"EXPERIMENT: {TARGET_CONCEPT} ({TARGET_TYPE}, {STEERING_MODE})")
print("=" * 70)
print(f"BETA per subspace : {BETA}")
print(f"Step range        : {STEP_RANGE}")
print(f"CLIP cap          : {CLIP_CAP}")
print(f"Diverse pairs     : {len(DIVERSE_PROMPT_PAIRS)}")
print(f"Eval seeds        : {len(EVAL_SEEDS)}")
print("=" * 70)

# ============================================================================
# CELL 8: LEARN STEERING VECTORS
# ============================================================================

vector_path = os.path.join(VECTOR_DIR, f"{TARGET_CONCEPT}_{STEERING_MODE}_diverse_vectors.pt")
if os.path.exists(vector_path):
    print(f"Loading cached vectors from {vector_path}")
    vectors = steerer.load_vectors(vector_path)
else:
    print(f"\nLearning vectors for {TARGET_CONCEPT}...")
    vectors = steerer.learn_vectors_diverse(
        prompt_pairs=DIVERSE_PROMPT_PAIRS, seed=0,
        top_k=TOP_K_VECTORS, verbose=True)
    steerer.save_vectors(vectors, vector_path)

# ============================================================================
# CELL 8B: QUICK STEERING TEST
# ============================================================================

DIAG_PROMPT = f"A {TARGET_CONCEPT.replace('_', ' ')} image in Van Gogh style."
DIAG_SEED = 188

print("=" * 70)
print(f"QUICK TEST: {TARGET_CONCEPT} ({STEERING_MODE})")
print("=" * 70)
print(f"Prompt: '{DIAG_PROMPT}'")

print("\nGenerating baseline (no steering)...")
baseline_img = steerer.generate(DIAG_PROMPT, DIAG_SEED, vectors=None)

# Sweep beta strengths. If diagnostic showed e.g. pooled_clipL dominates,
# the user should adjust these configs to vary just pooled_clipL.
if STEERING_MODE == "pincer_perstep":
    # OBJECT: identity is redundant + needs intervention on the post-MLP
    # modulation signal (TRACE App. D.1). Configs add tte_out alongside
    # the previous joint multi-subspace recipes.
    configs = [
        # tte_out-ONLY tests. If TRACE's finding holds, even a single-
        # subspace tte_out at moderate beta should remove the dog where
        # previous single-subspace failed.
        ("tte_only_mid",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 10.0,
          "ctx_clip": 0.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), None),
        ("tte_only_high",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 20.0,
          "ctx_clip": 0.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), None),

        # Joint recipes WITH tte_out. The new full-coverage approach.
        ("joint_with_tte_low",
         {"pooled_clipL": 3.0, "pooled_clipG": 6.0, "tte_out": 8.0,
          "ctx_clip": 6.0, "ctx_t5": 2.0},
         True, (0, N_STEPS), None),
        ("joint_with_tte_mid",
         {"pooled_clipL": 5.0, "pooled_clipG": 10.0, "tte_out": 12.0,
          "ctx_clip": 10.0, "ctx_t5": 2.0},
         True, (0, N_STEPS), None),
        ("joint_with_tte_high",
         {"pooled_clipL": 7.0, "pooled_clipG": 14.0, "tte_out": 18.0,
          "ctx_clip": 14.0, "ctx_t5": 3.0},
         True, (0, N_STEPS), None),

        # Old joint recipe (NO tte_out) for comparison. Should fail like before.
        ("old_joint_no_tte",
         {"pooled_clipL": 6.0, "pooled_clipG": 12.0, "tte_out": 0.0,
          "ctx_clip": 12.0, "ctx_t5": 3.0},
         True, (0, N_STEPS), None),
    ]
else:
    # STYLE: ctx_clip is the primary lever. Adding tte_out as a secondary
    # lever (TRACE finding); test whether including it helps style as well
    # as object.
    configs = [
        # ctx_clip-only (the previous recipe) at increasing strength.
        ("clip_only_low",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 0.0,
          "ctx_clip": 4.0,  "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),
        ("clip_only_mid",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 0.0,
          "ctx_clip": 8.0,  "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),
        ("clip_only_high",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 0.0,
          "ctx_clip": 12.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),

        # NEW: ctx_clip + tte_out (the TRACE-aligned style recipe).
        ("clip_plus_tte_low",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 3.0,
          "ctx_clip": 6.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),
        ("clip_plus_tte_mid",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 5.0,
          "ctx_clip": 8.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),

        # Sanity: tte_out alone for style.
        ("tte_only_style",
         {"pooled_clipL": 0.0, "pooled_clipG": 0.0, "tte_out": 8.0,
          "ctx_clip": 0.0, "ctx_t5": 0.0},
         True, (0, N_STEPS), 1.0),
    ]

test_images = []
for label, beta_val, clip_neg, srng, ccap in configs:
    print(f"  Generating: {label}... beta={beta_val}")
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
plt.suptitle(f"SD3.5 Steering Test ({STEERING_MODE})", fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 9: SINGLE-TARGET FULL UA/IRA/CRA
# ============================================================================

CLIP_NEGATIVE = True
print("\nRunning single-target UnlearnCanvas evaluation...")
eval_results = evaluator.evaluate_unlearning(
    steerer=steerer, vectors=vectors,
    target_concept=TARGET_CONCEPT, target_type=TARGET_TYPE,
    beta=BETA, clip_negative=CLIP_NEGATIVE,
    step_range=STEP_RANGE, clip_cap=CLIP_CAP,
    eval_seeds=EVAL_SEEDS, save_images=True, output_dir=OUTPUT_DIR)

print("\n" + "=" * 70)
print(f"{TARGET_CONCEPT}: UA={eval_results['UA']:.2%} "
      f"IRA={eval_results['IRA']:.2%} CRA={eval_results['CRA']:.2%}")
