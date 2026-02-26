# FLUX Architecture: Complete Data Flow & Steering Strategy Analysis

## Table of Contents
1. [High-Level Architecture Overview](#1-high-level-architecture-overview)
2. [Text Encoding Pipeline](#2-text-encoding-pipeline)
3. [Latent Space & VAE](#3-latent-space--vae)
4. [Timestep & Guidance Conditioning](#4-timestep--guidance-conditioning)
5. [Embedding Layers (Pre-Processing)](#5-embedding-layers-pre-processing)
6. [Double-Stream Blocks (MMDiT) — 19 Blocks](#6-double-stream-blocks-mmdit--19-blocks)
7. [Single-Stream Blocks — 38 Blocks](#7-single-stream-blocks--38-blocks)
8. [Output Projection & VAE Decoding](#8-output-projection--vae-decoding)
9. [Complete End-to-End Data Flow](#9-complete-end-to-end-data-flow)
10. [Steering Strategy: Where We Intervene](#10-steering-strategy-where-we-intervene)
11. [Potential Issues in Our Steering Strategy](#11-potential-issues-in-our-steering-strategy)

---

## 1. High-Level Architecture Overview

FLUX.1 is a **12-billion-parameter rectified flow transformer** that generates images from text descriptions. Unlike earlier diffusion models (Stable Diffusion 1.5/XL) which used a U-Net backbone, FLUX uses a pure transformer architecture with a novel two-stage design:

```
                        ┌─────────────────────┐
                        │    Text Prompt       │
                        └──────┬──────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌─────────────────┐             ┌─────────────────────┐
    │   CLIP Text      │             │  T5-XXL Text        │
    │   Encoder        │             │  Encoder            │
    │   (clip-vit-L)   │             │  (t5-v1_1-xxl)      │
    └────────┬────────┘             └──────────┬──────────┘
             │                                 │
     pooled_output (768)              sequence_output (B, S, 4096)
             │                                 │
             ▼                                 ▼
    ┌─────────────────┐             ┌─────────────────────┐
    │ time_text_embed  │             │  context_embedder   │
    │ MLP → (3072)     │             │  Linear(4096→3072)  │
    └────────┬────────┘             └──────────┬──────────┘
             │                                 │
             │  vec (B, 3072)                  │  txt_hidden (B, S, 3072)
             │  [global conditioning]          │  [per-token text features]
             │                                 │
             │         ┌──────────────┐        │
             │         │ Noise Latent │        │
             │         │  (B, C, H, W)│        │
             │         └──────┬───────┘        │
             │                │                │
             │         Pack into 2x2 patches   │
             │         + img_in Linear         │
             │                │                │
             │         img_hidden              │
             │         (B, N_img, 3072)        │
             ▼                ▼                ▼
    ┌──────────────────────────────────────────────┐
    │         19 DOUBLE-STREAM BLOCKS (MMDiT)      │
    │                                              │
    │   Image stream ◄──── Joint Attention ────► Text stream  │
    │   (separate weights)  (Q,K,V concat)   (separate weights)│
    │   + AdaLN-Zero (conditioned by vec)                      │
    └──────────────┬──────────────────┬────────────┘
                   │                  │
            img_hidden          txt_hidden
            (B, N_img, 3072)    (B, S, 3072)
                   │                  │
                   └───── CONCAT ─────┘
                          │
                   (B, N_img+S, 3072)
                          ▼
    ┌──────────────────────────────────────────────┐
    │         38 SINGLE-STREAM BLOCKS              │
    │                                              │
    │   Concatenated [img | txt] processed         │
    │   with SHARED weights                        │
    │   + AdaLN-Zero (conditioned by vec)          │
    └──────────────────────┬───────────────────────┘
                           │
                    (B, N_img+S, 3072)
                           │
                   Extract image tokens only
                   (discard text tokens)
                           │
                    (B, N_img, 3072)
                           ▼
    ┌──────────────────────────────────────────────┐
    │         Final Layer (LastLayer)               │
    │   AdaLN → Linear(3072 → 64)                  │
    └──────────────────────┬───────────────────────┘
                           │
                    Unpack 2x2 patches
                    (B, 16, H, W)
                           ▼
    ┌──────────────────────────────────────────────┐
    │         VAE Decoder (16-channel)             │
    │   Latent → Pixel Image                       │
    └──────────────────────┬───────────────────────┘
                           │
                    Output Image (H×8, W×8)
```

**Key Numbers:**
- **Hidden size:** 3072 (= 24 attention heads x 128 head dim)
- **Double-stream blocks:** 19 (separate weights for text/image)
- **Single-stream blocks:** 38 (shared weights for concatenated text+image)
- **VAE channels:** 16 (4x more than SDXL's 4 channels)
- **T5 max sequence length:** 512 tokens
- **Patch size:** 2x2 (latent pixels packed into patches)

---

## 2. Text Encoding Pipeline

FLUX uses **dual text encoders** in parallel — this is critical for understanding steering:

### 2a. CLIP Text Encoder (clip-vit-large-patch14)

```
Text Prompt
    │
    ▼
CLIPTokenizer (max_length=77)
    │
    ▼
CLIPTextModel
    │
    ├── last_hidden_state: (B, 77, 768) — NOT USED by FLUX
    └── pooler_output:     (B, 768)     — USED as pooled_prompt_embeds
```

**Role:** Provides a **single global vector** (768-dim) summarizing the prompt. This gets fused with the timestep embedding to produce the `vec` conditioning signal that modulates EVERY transformer block via AdaLN.

**Implications for steering:** CLIP captures high-level semantic concepts (style, mood, composition). Styles like "Van Gogh" or "Watercolor" are strongly encoded here. This is why `time_text_embed` is an effective steering target for style unlearning.

### 2b. T5-XXL Text Encoder (google/t5-v1_1-xxl)

```
Text Prompt
    │
    ▼
T5TokenizerFast (max_length=512, padding="max_length")
    │
    ├── input_ids:      (B, 512)
    └── attention_mask:  (B, 512)  — 1=real token, 0=padding
    │
    ▼
T5EncoderModel
    │
    └── last_hidden_state: (B, 512, 4096) — USED as prompt_embeds
```

**Role:** Provides **per-token dense embeddings** (4096-dim for each of up to 512 tokens). T5 captures fine-grained language understanding — specific objects, spatial relationships, detailed descriptions.

**Critical detail:** With `max_length=512` and `padding="max_length"`, most prompts produce many padding tokens. A 10-word prompt might have ~15 real tokens and ~497 padding tokens. **The attention mask is essential** for any pooling operation — averaging without the mask includes garbage padding values.

### 2c. Text ID Tensor

```python
text_ids = torch.zeros(prompt_embeds.shape[1], 3)  # (512, 3) of zeros
```

A positional ID tensor for the text stream. All zeros — text tokens have no spatial position (unlike image tokens which encode height/width). Used for RoPE in the attention mechanism.

---

## 3. Latent Space & VAE

### 3a. VAE Architecture

FLUX uses a **16-channel VAE** (AutoencoderKL), compared to SDXL's 4-channel VAE:

```
Image (B, 3, H, W)  in pixel space
    │
    ▼
VAE Encoder (8x spatial compression)
    │
    ▼
Latent (B, 16, H/8, W/8)
    │
    ├── scaling_factor applied
    └── shift_factor applied
```

**16 channels vs 4 channels:** More channels allow the VAE to capture finer details — richer textures, more accurate color gradients, sharper details. This is one reason FLUX produces higher-quality images than SD1.5/XL.

### 3b. Latent Packing (2x2 patches)

This is unique to FLUX and critical for understanding sequence lengths:

```python
# Before packing: (B, 16, H/8, W/8) = (B, 16, h, w) where h,w are latent dims
# After packing:  (B, (h/2)*(w/2), 16*4) = (B, N_img, 64)

# Example: 1024x1024 image
#   Latent: (1, 16, 128, 128)
#   Packed: (1, 4096, 64)        ← N_img = 4096 tokens

# Example: 512x512 image
#   Latent: (1, 16, 64, 64)
#   Packed: (1, 1024, 64)        ← N_img = 1024 tokens
```

The packing operation reshapes 2x2 spatial patches into single tokens:
```python
latents = latents.view(B, C, h//2, 2, w//2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)       # (B, h/2, w/2, C, 2, 2)
latents = latents.reshape(B, (h//2)*(w//2), C*4)   # (B, N_img, 64)
```

### 3c. Latent Image IDs (Spatial Position Encoding)

```python
latent_image_ids = torch.zeros(h//2, w//2, 3)
latent_image_ids[..., 1] += torch.arange(h//2)[:, None]  # row index
latent_image_ids[..., 2] += torch.arange(w//2)[None, :]  # col index
# Result: (N_img, 3) — each patch has (0, row, col) position
```

These IDs feed into **Rotary Positional Embeddings (RoPE)** applied to Q and K in every attention layer. This is how FLUX understands spatial layout without learned position embeddings, enabling resolution flexibility.

---

## 4. Timestep & Guidance Conditioning

### 4a. Timestep Embedding

The diffusion timestep `t` (a scalar in [0, 1]) is embedded via sinusoidal projection:

```
timestep (scalar)
    │
    ▼
Sinusoidal Embedding (256-dim)
    │
    ▼
time_in: MLPEmbedder(256 → 3072)
    │ (SiLU + Linear)
    ▼
time_emb: (B, 3072)
```

### 4b. Guidance Embedding (FLUX-dev / FLUX-pro only)

For guidance-distilled models, the guidance scale is also embedded:

```
guidance_scale (scalar, e.g. 3.5)
    │
    ▼
Sinusoidal Embedding (256-dim)
    │
    ▼
guidance_in: MLPEmbedder(256 → 3072)
    │
    ▼
guidance_emb: (B, 3072)
```

**Note:** FLUX-schnell (4-step) does NOT use guidance embedding (`guidance_embeds=False`).

### 4c. Pooled Text Embedding (CLIP)

```
CLIP pooler_output: (B, 768)
    │
    ▼
text_embedder: MLPEmbedder(768 → 3072)
    │ (SiLU + Linear)
    ▼
text_emb: (B, 3072)
```

### 4d. Combined Conditioning Vector (`vec`)

All conditioning signals are summed into a single vector that drives AdaLN in every block:

```python
vec = time_emb + text_emb                    # Always: timestep + CLIP pooled text
if guidance_embeds:
    vec = vec + guidance_emb                  # FLUX-dev/pro: add guidance too
# vec shape: (B, 3072)
```

**This is the `time_text_embed` module in our steering code.** It fuses timestep + CLIP pooled text (+ guidance) into the global conditioning signal.

---

## 5. Embedding Layers (Pre-Processing)

Before entering the transformer blocks, all inputs are projected to the shared hidden dimension (3072):

### 5a. Image Input Projection

```python
img_in = nn.Linear(64, 3072)  # packed latent dim → hidden size
# Input: (B, N_img, 64) packed latents
# Output: (B, N_img, 3072) image hidden states
```

### 5b. Text Input Projection (context_embedder)

```python
context_embedder = nn.Linear(4096, 3072)  # T5 dim → hidden size
# Input: (B, 512, 4096) T5 encoder output
# Output: (B, 512, 3072) text hidden states
```

**This is the first steering target — `context_embedder`.** It is the ONLY point where T5 text information enters the transformer. Steering here intercepts ALL text content before it interacts with image tokens.

### 5c. Summary of Pre-Processing

```
                 ┌─────────────┐
Packed Latents ──┤   img_in    ├──► img_hidden: (B, N_img, 3072)
  (B, N_img, 64) │ Linear      │
                 └─────────────┘

                 ┌──────────────────┐
T5 Embeddings ───┤ context_embedder ├──► txt_hidden: (B, 512, 3072)  ◄── STEERING TARGET
  (B, 512, 4096) │ Linear           │
                 └──────────────────┘

                 ┌──────────────────┐
Timestep+CLIP ───┤ time_text_embed  ├──► vec: (B, 3072)             ◄── STEERING TARGET
  (sinusoidal+   │ MLPs + sum       │    [global conditioning for AdaLN]
   pooled text)  └──────────────────┘
```

---

## 6. Double-Stream Blocks (MMDiT) — 19 Blocks

These are the core of FLUX's multimodal processing. Each block maintains **separate streams** for image and text, joining them **only for attention**.

### 6a. Block Structure

```
                    img_hidden                      txt_hidden
                    (B, N_img, 3072)                (B, 512, 3072)
                         │                               │
                    ┌────┴────┐                     ┌────┴────┐
                    │ AdaLN   │ ◄── vec (B,3072)    │ AdaLN   │ ◄── vec (B,3072)
                    │ (img)   │     [6 params:      │ (txt)   │     [6 params:
                    │         │      shift,scale,    │         │      shift,scale,
                    │         │      gate for attn   │         │      gate for attn
                    │         │      & MLP]          │         │      & MLP]
                    └────┬────┘                     └────┬────┘
                         │                               │
    ┌────────────────────┤                               ├────────────────────┐
    │ Image Q,K,V        │                               │ Text Q,K,V        │
    │ to_q(img)→Q_img    │                               │ add_q_proj(txt)→Q_txt │ ◄── STEERING
    │ to_k(img)→K_img    │                               │ add_k_proj(txt)→K_txt │ ◄── STEERING
    │ to_v(img)→V_img    │                               │ add_v_proj(txt)→V_txt │ ◄── STEERING
    └────────┬───────────┘                               └────────┬───────────┘
             │                                                     │
             │              ┌─────────────────────────┐            │
             │              │    JOINT ATTENTION       │            │
             │              │                         │            │
             └──────────────┤  Q = [Q_img ; Q_txt]    ├────────────┘
                            │  K = [K_img ; K_txt]    │
                            │  V = [V_img ; V_txt]    │
                            │                         │
                            │  + RoPE on Q,K          │
                            │    (using img_ids &     │
                            │     txt_ids for pos)    │
                            │                         │
                            │  Attn = softmax(QK^T/√d)│
                            │  Out = Attn × V         │
                            └────────────┬────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                        img_attn_out           txt_attn_out
                              │                     │
                    ┌─────────┴────────┐  ┌─────────┴────────┐
                    │   to_out[0]      │  │  to_add_out      │
                    │ Linear(3072→3072)│  │ Linear(3072→3072) │
                    └─────────┬────────┘  └─────────┬────────┘
                              │ ◄── STEERING               │
                    Gate (α₁) │                    Gate (α₁) │
                    Residual  +                    Residual  +
                              │                              │
                    ┌─────────┴────────┐  ┌─────────┴────────┐
                    │   AdaLN (MLP)    │  │   AdaLN (MLP)    │
                    │   modulate       │  │   modulate       │
                    │   Feed-Forward   │  │   Feed-Forward   │
                    │   Gate + Residual│  │   Gate + Residual│
                    └─────────┬────────┘  └─────────┬────────┘
                              │                              │
                        img_hidden'                    txt_hidden'
                        (B, N_img, 3072)               (B, 512, 3072)
```

### 6b. AdaLN-Zero Modulation (Detail)

Each block produces **6 modulation parameters** from `vec` for each stream (12 total per block):

```python
# adaLN_modulation: Linear(3072 → 6*3072) applied to vec
modulation_output = silu(vec) → Linear → chunk into 6 vectors

shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
# Each: (B, 3072)
```

The modulation is applied as:
```python
# Pre-attention:
x_normed = LayerNorm(x)
x_modulated = x_normed * (1 + scale_msa) + shift_msa   # Affine transform
# → feeds into QKV projection

# Post-attention residual:
x = x + gate_msa * attn_output                          # Gated residual

# Pre-MLP:
x_normed = LayerNorm(x)
x_modulated = x_normed * (1 + scale_mlp) + shift_mlp

# Post-MLP residual:
x = x + gate_mlp * mlp_output
```

**Key insight for steering:** `vec` (from `time_text_embed`) controls AdaLN in ALL 19 double-stream blocks AND all 38 single-stream blocks. It is a powerful global lever. Styles flow heavily through this path because AdaLN modulates *how* features are processed rather than *what* features are present.

### 6c. Joint Attention Mechanism (Detail)

The joint attention in FLUX is where text and image truly interact:

```python
# SEPARATE projections (different weight matrices for each modality)
Q_img = to_q(img_normed)              # (B, N_img, 3072)
K_img = to_k(img_normed)              # (B, N_img, 3072)
V_img = to_v(img_normed)              # (B, N_img, 3072)

Q_txt = add_q_proj(txt_normed)        # (B, 512, 3072) ◄── STEERING TARGET
K_txt = add_k_proj(txt_normed)        # (B, 512, 3072) ◄── STEERING TARGET
V_txt = add_v_proj(txt_normed)        # (B, 512, 3072) ◄── STEERING TARGET

# Reshape for multi-head attention (24 heads, 128 dim each)
# Apply RoPE to Q and K using positional IDs
Q_img, K_img = apply_rope(Q_img, K_img, img_ids)
Q_txt, K_txt = apply_rope(Q_txt, K_txt, txt_ids)

# CONCATENATE for joint attention
Q = concat([Q_img, Q_txt], dim=1)     # (B, N_img+512, 3072)
K = concat([K_img, K_txt], dim=1)     # (B, N_img+512, 3072)
V = concat([V_img, V_txt], dim=1)     # (B, N_img+512, 3072)

# Standard scaled dot-product attention
attn_output = scaled_dot_product_attention(Q, K, V)
# Result: (B, N_img+512, 3072)

# SPLIT back into modalities
img_attn_output = attn_output[:, :N_img, :]    # (B, N_img, 3072)
txt_attn_output = attn_output[:, N_img:, :]    # (B, 512, 3072)

# SEPARATE output projections
img_out = to_out[0](img_attn_output)           # (B, N_img, 3072) ◄── STEERING TARGET
txt_out = to_add_out(txt_attn_output)          # (B, 512, 3072)
```

**What this means for concept flow:**
- **Text Keys** (`add_k_proj`): Determine WHAT text concepts image tokens can attend to. Steering K makes the concept "invisible" to image attention.
- **Text Queries** (`add_q_proj`): Determine HOW text tokens query for relevant image features. Steering Q prevents the concept from finding matching image features.
- **Text Values** (`add_v_proj`): Carry the actual CONTENT that gets transferred to image tokens through attention. Steering V prevents concept content from reaching images.
- **Image Output** (`to_out[0]`): The image representation AFTER it has attended to text. This is the **FLUX analog of SD's cross-attention output** that CASteer targets. Steering here removes concept influence from the post-attention image features.

### 6d. How Concepts Flow Through Double-Stream Blocks

Understanding concept flow is essential for effective steering:

```
STYLE CONCEPTS (e.g., "Van Gogh"):
  ╔════════════════════════════════════════════════╗
  ║ Styles flow PRIMARILY through the AdaLN path: ║
  ║                                                ║
  ║ CLIP pooler → time_text_embed → vec → AdaLN   ║
  ║                                                ║
  ║ AdaLN modulates HOW all features are processed ║
  ║ (scale, shift, gate parameters). This is why   ║
  ║ styles affect the entire image globally.        ║
  ║                                                ║
  ║ ALSO through attention (secondary path):        ║
  ║ T5 embeddings → add_k/add_q → joint attention  ║
  ╚════════════════════════════════════════════════╝

OBJECT CONCEPTS (e.g., "Dog"):
  ╔════════════════════════════════════════════════╗
  ║ Objects flow PRIMARILY through attention:      ║
  ║                                                ║
  ║ T5 embeddings → add_k/add_q/add_v →           ║
  ║   joint attention → to_out[0]                  ║
  ║                                                ║
  ║ Objects are spatially localized — they appear  ║
  ║ in specific image regions through attention    ║
  ║ weights that "write" object features into      ║
  ║ those spatial locations.                       ║
  ║                                                ║
  ║ AdaLN has minimal object-specific influence    ║
  ║ because vec is a GLOBAL signal, not spatial.   ║
  ╚════════════════════════════════════════════════╝
```

---

## 7. Single-Stream Blocks — 38 Blocks

After the 19 double-stream blocks, the image and text hidden states are **concatenated** and processed jointly through 38 single-stream blocks with **shared weights**.

### 7a. Concatenation

```python
# After 19 double-stream blocks:
img_hidden: (B, N_img, 3072)
txt_hidden: (B, 512, 3072)

# Concatenate:
hidden = concat([img_hidden, txt_hidden], dim=1)
# hidden: (B, N_img + 512, 3072)
```

### 7b. Single-Stream Block Structure

```
    hidden_states: (B, N_img+512, 3072)
         │
    ┌────┴────────────────────────────────────┐
    │ AdaLN Modulation ◄── vec (B, 3072)      │
    │ (3 params: scale, shift, gate)          │
    └────┬────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────────┐
    │ QKV Projection (SHARED for both text    │
    │ and image — single set of weights)      │
    │                                          │
    │ qkv = linear(modulated_hidden)           │
    │ q, k, v = split(qkv)                    │
    │ + RoPE on q, k                           │
    │ + Attention (self-attention over full    │
    │   concatenated sequence)                 │
    └────┬────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────────┐
    │ MLP (parallel with attention in ViT-22B │
    │ style — computed in parallel, then      │
    │ concatenated with attention output)      │
    └────┬────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────────┐
    │ proj_out: Linear(concatenated → 3072)   │ ◄── STEERING TARGET
    │ Gate + Residual connection              │
    └────┬────────────────────────────────────┘
         │
    hidden_states': (B, N_img+512, 3072)
```

### 7c. Key Differences from Double-Stream

| Property | Double-Stream (19 blocks) | Single-Stream (38 blocks) |
|----------|--------------------------|---------------------------|
| Weights | Separate for text/image | Shared for both |
| QKV | Separate projections per modality | Single projection for all |
| Input | Separate streams | Concatenated sequence |
| AdaLN | 6 params per stream (12 total) | 3 params for joint sequence |
| Output | Two separate streams | One joint sequence |
| Parameter share | ~54% of model params | ~46% of model params |

### 7d. Text/Image Token Layout in Single-Stream

```
Position: [0, 1, ..., N_img-1, N_img, N_img+1, ..., N_img+511]
Content:  [  IMAGE TOKENS     |         TEXT TOKENS              ]
```

**Critical for steering:** When hooking `proj_out` in single-stream blocks, the output contains BOTH image and text tokens concatenated. To steer only one modality, you must slice:
- Image tokens: `output[:, :N_img, :]`
- Text tokens: `output[:, N_img:, :]`

### 7e. Why 38 Single-Stream Blocks?

The 2:1 ratio (38 single-stream vs 19 double-stream) is a design choice by BFL. The rationale (inferred):
- Double-stream blocks establish modality-specific representations and cross-modal attention patterns
- Single-stream blocks then deeply fuse these representations through shared processing
- More fusion layers allow image tokens to thoroughly absorb and refine information from text tokens

In FLUX.2, this ratio shifted to 48:8, emphasizing even more single-stream processing, suggesting the fusion stage is where most of the "generation work" happens.

---

## 8. Output Projection & VAE Decoding

### 8a. Extract Image Tokens

After single-stream processing, only image tokens are kept:

```python
hidden = hidden[:, :N_img, :]   # Discard text tokens
# hidden: (B, N_img, 3072)
```

### 8b. Final Layer

```python
# AdaLN conditioned by vec, then project to output channels
final_out = final_layer(hidden, vec)
# Linear(3072 → 64)  — maps back to packed latent dimension
# Output: (B, N_img, 64)
```

### 8c. Unpack 2x2 Patches

```python
# Reverse the packing: (B, N_img, 64) → (B, 16, H/8, W/8)
latents = latents.view(B, h//2, w//2, 16, 2, 2)
latents = latents.permute(0, 3, 1, 4, 2, 5)
latents = latents.reshape(B, 16, h, w)
```

### 8d. VAE Decoding

```python
# Apply inverse scaling and shift
latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

# Decode to pixel space
image = vae.decode(latents)
# Output: (B, 3, H, W) — full-resolution RGB image
```

---

## 9. Complete End-to-End Data Flow

Here is the complete pipeline flow during a single denoising step:

```
STEP-BY-STEP FLOW FOR ONE DENOISING STEP:
═══════════════════════════════════════════

1. TEXT ENCODING (done once, reused across all steps):
   ┌──────────────────────────────────────────────────────┐
   │ prompt ──► CLIP ──► pooled_prompt_embeds (B, 768)    │
   │ prompt ──► T5   ──► prompt_embeds (B, 512, 4096)     │
   │ text_ids = zeros(512, 3)                             │
   └──────────────────────────────────────────────────────┘

2. LATENT PREPARATION (done once):
   ┌──────────────────────────────────────────────────────┐
   │ random noise ──► pack 2x2 ──► latents (B, N_img, 64)│
   │ latent_image_ids: (N_img, 3) with (0, row, col)     │
   └──────────────────────────────────────────────────────┘

3. TIMESTEP SCHEDULE:
   ┌──────────────────────────────────────────────────────┐
   │ sigmas = linspace(1.0, 1/N, N) shifted by mu        │
   │ For Schnell: N=4 steps                               │
   │ For Dev: N=28 steps (default)                        │
   └──────────────────────────────────────────────────────┘

4. FOR EACH DENOISING STEP t:
   ┌──────────────────────────────────────────────────────┐
   │ a) CONDITIONING:                                     │
   │    timestep = t / 1000                               │
   │    guidance = full(guidance_scale)  [if dev/pro]     │
   │                                                      │
   │ b) TRANSFORMER FORWARD PASS:                         │
   │    transformer(                                      │
   │      hidden_states = latents,      ◄── packed noise  │
   │      timestep = timestep/1000,     ◄── normalized t  │
   │      guidance = guidance,          ◄── CFG scale     │
   │      pooled_projections = pooled_prompt_embeds,      │
   │      encoder_hidden_states = prompt_embeds, ◄── T5   │
   │      txt_ids = text_ids,           ◄── zeros         │
   │      img_ids = latent_image_ids,   ◄── spatial pos   │
   │    )                                                 │
   │                                                      │
   │    INSIDE THE TRANSFORMER:                           │
   │    ┌──────────────────────────────────────────┐      │
   │    │ i.   img_in(hidden_states) → img_hidden  │      │
   │    │ ii.  context_embedder(encoder_hs)→txt_h  │ ◄STR │
   │    │ iii. time_text_embed(t, clip) → vec      │ ◄STR │
   │    │ iv.  FOR block in 19 double_stream:      │      │
   │    │        img_h, txt_h = block(             │      │
   │    │          img_h, txt_h, vec,              │      │
   │    │          img_ids, txt_ids)               │      │
   │    │        [joint attn with separate Q,K,V]  │ ◄STR │
   │    │ v.   concat → hidden = [img_h ; txt_h]  │      │
   │    │ vi.  FOR block in 38 single_stream:      │      │
   │    │        hidden = block(hidden, vec)        │ ◄STR │
   │    │ vii. Extract img tokens, final_layer     │      │
   │    └──────────────────────────────────────────┘      │
   │                                                      │
   │ c) SCHEDULER STEP:                                   │
   │    noise_pred = transformer output                   │
   │    latents = scheduler.step(noise_pred, t, latents)  │
   │    [Euler discrete step for flow matching]           │
   └──────────────────────────────────────────────────────┘

5. DECODE:
   ┌──────────────────────────────────────────────────────┐
   │ unpack latents → VAE decode → postprocess → PIL      │
   └──────────────────────────────────────────────────────┘
```

---

## 10. Steering Strategy: Where We Intervene

Our steering strategy intercepts activations at specific points in the architecture and subtracts the projection onto the concept direction:

```
Steering formula: output' = output - beta * clamp(output . d, min=0) * d

Where:
  d = learned concept direction (unit vector in 3072-dim space)
  beta = steering strength (scalar)
  clamp(., min=0) = only remove positive projections (CASteer method)
```

### 10a. Mode Summary vs Architecture Location

```
ARCHITECTURE LOCATION          STEERING MODES THAT TARGET IT
─────────────────────          ─────────────────────────────
context_embedder               entry_point, hybrid, all, object_v2
  └─ T5→3072 projection       [Text bottleneck: ALL T5 info passes through here]

time_text_embed                entry_point, hybrid, all, object_v2
  └─ timestep+CLIP→3072 MLP   [Global conditioning: styles flow heavily here]

Double-Stream Blocks (×19):
  add_q_proj                   double_proj, hybrid, all
  add_k_proj                   double_proj, hybrid, all
    └─ Text Q/K projections    [Controls what text attends to / is attended by]
  add_v_proj                   object
    └─ Text Value projection   [Carries text content to image through attention]
  to_out[0]                    block, all, object, object_v2, joint_attn
    └─ Image attn output proj  [CASteer cross-attention analog: post-attention image]

Single-Stream Blocks (×38):
  proj_out (text slice)        block
    └─ Text portion of output  [Text representation after shared processing]
  proj_out (image slice)       object, object_v2
    └─ Image portion of output [Image tokens that absorbed object info from text]
```

### 10b. Learning Approach

**For styles (single prompt pair + multiple seeds):**
```
pos_prompt = "Van Gogh style"
neg_prompt = "neutral style"
seeds = [0, 1, 2, ..., 19]

For each seed:
  act_pos = run_pipeline(pos_prompt, seed)  → collect hook activations
  act_neg = run_pipeline(neg_prompt, seed)  → collect hook activations
  diff += (act_pos - act_neg)

direction = normalize(diff / n_seeds)
```

**For objects (diverse prompt pairs + single seed) — CASteer methodology:**
```
prompt_pairs = [
  ("tench with Dog", "tench"),
  ("goldfish with Dog", "goldfish"),
  ("great white shark with Dog", "great white shark"),
  ... (50 pairs total using ImageNet classes)
]

For each (pos, neg) pair:
  act_pos = run_pipeline(pos, seed=0) → collect
  act_neg = run_pipeline(neg, seed=0) → collect
  diff += (act_pos - act_neg)

direction = normalize(diff / n_pairs)
```

The diverse prompts make the object concept direction **context-invariant** — the varying contexts cancel out, leaving only the "Dog-ness" direction.

---

## 11. Potential Issues in Our Steering Strategy

### Issue 1: AdaLN Modulation Bypass

**Problem:** Steering at `context_embedder` and `time_text_embed` catches text information at the entry points, but misses how AdaLN propagates this information. The `vec` conditioning signal modulates EVERY block (57 total) through 6 parameters each:

```
vec → AdaLN → scale/shift/gate for attention AND MLP in EVERY block
```

Steering `time_text_embed` output changes `vec` before it enters the blocks, but `vec` is used **as-is** in every block — there is no per-block processing that we could intercept. However, the concept information encoded in `vec` gets "baked into" the scale/shift/gate parameters, which then modulate activations in complex nonlinear ways. Projection removal on a 3072-dim vector may not adequately neutralize the concept's influence through 342 modulation parameters (57 blocks x 6 params).

**Potential fix:** Instead of steering `vec` directly, consider steering the AdaLN output (the modulation parameters themselves) at each block where the concept shows strongest signal.

### Issue 2: RoPE Entanglement in Attention

**Problem:** We steer `add_k_proj` and `add_q_proj` outputs, but RoPE is applied AFTER these projections:

```
K_txt = add_k_proj(txt_normed)     ◄── We steer here
K_txt = apply_rope(K_txt, txt_ids) ◄── RoPE rotates the vector AFTER steering
```

RoPE applies rotation matrices that mix dimensions in a position-dependent way. The concept direction we learned (pre-RoPE) may not align with the actual concept direction after RoPE rotation. This could reduce steering effectiveness at `add_k_proj` and `add_q_proj`.

**Potential fix:** Learn steering vectors POST-RoPE (from the attention Q/K after rotation), or steer at the attention output instead (which is post-RoPE by definition).

### Issue 3: 4-Step Schnell vs. 50-Step Diffusion

**Problem:** FLUX-schnell uses only 4 denoising steps, while CASteer was designed for SD1.5 with 50 steps. Our per-step steering vectors have only 4 opportunities to intervene. With top_k=15 selection across (19+38) blocks x 4 steps = 228 candidates, we're only covering ~6.5% of possible intervention points.

**Implications:**
- Each step in Schnell makes a much larger "jump" in latent space than in 50-step models
- Steering too aggressively in any one step can cause artifacts
- Steering too weakly across 4 steps may not accumulate enough concept removal
- The optimal beta for 4-step (~2-5) needs to compensate for fewer intervention opportunities compared to 50-step models

**Potential fix:** Use keep-all vectors instead of top_k for Schnell. Or scale beta by (50/4) to compensate for fewer steps.

### Issue 4: Double-Stream `to_out[0]` — Image-Only or Joint?

**Problem:** In the double-stream blocks, `to_out[0]` processes the IMAGE portion of the attention output (after the joint attention split). But the "joint_attn" mode applies `mean(dim=(0,1))` pooling over the FULL output — which should only be image tokens (since to_out[0] receives the split image portion). However, verifying this assumption is critical.

If `to_out[0]` actually receives the full concatenated attention output (pre-split), then our pooling would average image and text features together, diluting the concept signal.

**Status:** The code correctly identifies `to_out[0]` as the image-stream output projection (separate from `to_add_out` for text). The assumption appears correct based on the HuggingFace Diffusers implementation.

### Issue 5: SingleStream Token Ordering Assumption

**Problem:** Our SingleStream hooks assume image tokens come first and text tokens come last: `[img_tokens | txt_tokens]`. This is based on the concatenation order in the pipeline:

```python
hidden = concat([img_hidden, txt_hidden], dim=1)
```

If any SingleStream block internally reorders tokens (unlikely but possible with some implementations), our image/text slicing would be wrong. The hooks extract:
- Image slice: `output[:, :N_img, :]`
- Text slice: `output[:, N_img:, :]`

This depends on `N_img` being correctly computed as `output.shape[1] - 512` (T5 max seq length). If the actual text length differs (e.g., due to truncation or different max_sequence_length), slicing would be incorrect.

### Issue 6: Concept Direction Orthogonality

**Problem:** Our projection removal assumes the concept direction `d` is orthogonal to all "useful" directions. But in a 3072-dimensional space with rich semantic structure, the concept direction may have significant overlap with non-concept features.

For example, "Van Gogh" direction might overlap with "thick brushstrokes" and "warm colors" — removing the Van Gogh direction could also dampen these general artistic features, reducing IRA (In-domain Retain Accuracy).

**Potential fix:** Use orthogonal projection against a subspace of "retain" concepts to ensure the steering vector is truly concept-specific. Or use the CASteer diverse-prompt approach (which naturally cancels non-concept features).

### Issue 7: The text_ids=zeros Problem for RoPE

**Problem:** Text tokens have `text_ids = zeros(512, 3)` — all zeros for position. This means RoPE gives ALL text tokens the SAME positional encoding, making them position-invariant. While this is by design (text has no spatial position), it means:

1. Text tokens cannot be distinguished by position in attention — only by content
2. Steering vectors learned from text projections are inherently position-agnostic
3. This is actually GOOD for steering (position doesn't confound the concept direction)

**Status:** Not a bug — this is correct behavior. Text position-invariance actually helps steering effectiveness.

### Issue 8: Norm Shrinkage Without Renormalization

**Problem:** In modes that steer many layers (e.g., "all" mode steers 40+ surfaces), the cumulative projection removal shrinks activation norms. After removing the projection onto `d` from output at every layer:

```
||output - beta * proj||  <  ||output||
```

Over many layers, norms shrink multiplicatively, potentially degrading image quality or making later-layer steering ineffective (small norms = small projections = diminishing returns).

**Status:** Partially addressed — object/object_v2 modes use renormalization. But hybrid/entry_point/block modes do NOT renormalize, which could be an issue for aggressive beta values.

---

## References

Architecture details synthesized from:
- [FLUX Model Architecture — DeepWiki](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)
- [Demystifying Flux Architecture (arXiv:2507.09595)](https://arxiv.org/abs/2507.09595)
- [Stable Diffusion 3 & FLUX: Complete Guide to MMDiT Architecture](https://blog.sotaaz.com/post/sd3-flux-architecture-en)
- [FluxTransformer2DModel — HuggingFace Diffusers Docs](https://huggingface.co/docs/diffusers/main/en/api/models/flux_transformer)
- [FreeFlux: Understanding Layer-Specific Roles in RoPE-Based MMDiT](https://arxiv.org/abs/2503.16153)
- [Flavors of Attention in Modern Diffusion Models — Sayak Paul](https://sayak.dev/posts/attn-diffusion.html)
- [Diffusers welcomes FLUX-2 — HuggingFace Blog](https://huggingface.co/blog/flux-2)
- [HuggingFace Diffusers FluxPipeline source code](https://github.com/huggingface/diffusers)
- CASteer: Steering Diffusion Models (reference paper in repository)
- UnlearnCanvas benchmark (reference paper in repository)
