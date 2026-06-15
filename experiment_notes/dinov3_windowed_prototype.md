# Verification Addendum — Read This First

> This report was produced by a multi-agent workflow and then **adversarially verified**:
> each correctness-critical claim was handed to an independent skeptic told to *refute* it
> against the real `transformers` DINOv3 source and the RF-DETR codebase.
> **Result: 12 claims held, 2 broke, 1 was unverifiable.** The corrections below override
> the body of the report where they conflict. The core design is sound; the corrections are
> about *how* the global-attention block must be implemented and two factual fixes.

## ❗ Correction 1 (BREAK) — the global-block merge is the one part you must not shortcut
The body (§3 steps 8–10, executive summary) says duplicated `[CLS,reg]` prefixes are "kept
unrotated via identity (cos=1,sin=0) rows, faithfully replicating DINOv2 semantics." That
wording is misleading:
- **Identity rows do NOT exist in stock `transformers`.** DINOv3's `cos`/`sin` have exactly
  `num_patches` rows; there are no prefix rows.
- **Stock `apply_rotary_pos_emb` will silently corrupt the merged sequence.** It assumes
  patches are a *contiguous trailing block* (`num_prefix = num_tokens − num_patches`). After
  the DINOv2-style window merge the layout is `[pfx, win0-patches, pfx, win1-patches, …]` —
  prefixes are interleaved, so the trailing-slice rotates the wrong tokens.

The design is only correct because §4.3/§4.4 **write a custom** full-length `cos`/`sin` (with
identity prefix rows) **and a custom position-indexed rotary apply** that replaces the stock
function. That custom path is **load-bearing — not optional.** Two implementation options,
both validated as correct:
- **Option A (recommended to prototype first — lowest risk):** apply RoPE *per-window, before
  merging* (windows still separate batch rows → stock trailing-slice apply is correct per
  window), then merge q/k/v for the global matmul. Avoids any interleaved-prefix RoPE entirely.
- **Option B (what the body codes):** custom position-indexed apply on the merged sequence
  with identity prefix rows folded through the *identical* merge as the tokens.

Prototype **A** first; fall back to B only if you don't want to restructure the layer.

## ❗ Correction 2 (BREAK) — factual error about patch_size=16 and weight loading
§5/§7 imply existing `patch_size=16` configs "confirm DINOv3 variants load from RF-DETR
finetuned checkpoints." **There is no DINOv3 in the repo** — `grep -rni dinov3 src/` is empty.
The true statement: current `RFDETRMediumConfig`/`RFDETRLargeConfig` use `patch_size=16`, which
trips `dinov2.py:115-122` (`load_dinov2_weights=False`) and so those **DINOv2** variants skip
`facebook/dinov2-*` hub weights. This says **nothing** about DINOv3. A DINOv3 backbone needs its
**own** `from_pretrained` path and will **not** flow through `dinov2.py`'s `load_dinov2_weights`
gate. Do not reason about DINOv3 weight-loading through that line.

## ⚠️ Correction 3 (UNCERTAIN) — the num_windows=1 equivalence is a test to write, not a proven fact
The "reduces exactly to upstream DINOv3ViTBackbone at num_windows=1" claim can't be verified
because the module doesn't exist yet. To make it *true and testable*: (1) port from
`DINOv3ViTBackbone`, **not** the DINOv2 windowed module (keep RoPE-in-attention, the float32
`maybe_autocast` angle wrapper, `scaling=head_dim**-0.5`); (2) replicate DINOv3's **asymmetric
q/k/v biases** (`key_bias=False`), **gated-MLP** option, and **per-stage LayerNorm policy**;
(3) gate the fold strictly on `num_windows>1`; (4) **force eager attention** on both sides when
asserting bit-equality (SDPA ≠ eager bitwise).

---

# Prototype Investigation Report: Windowed-Attention DINOv3 ViT Backbone for RF-DETR

**Status:** Feasibility prototype design. Verified against `transformers` DINOv3 source, the existing RF-DETR windowed-DINOv2 backbone, and a numerical RoPE-fold check (diff ≈ 0 / 3e-15) run in this environment.

---

## 1. Executive Summary

A windowed-attention DINOv3 ViT backbone is **feasible** and can be built by reusing RF-DETR's existing Swin-style batch-fold windowing scaffold verbatim while replacing DINOv2's learned absolute position-embedding (added once before the fold) with DINOv3's per-layer 2D-axial RoPE injected *inside* attention. The chosen design — **"batch-fold + RoPE-fold"** — folds the global RoPE `cos`/`sin` tables through the *identical* window-major reshape/permute used for patch tokens, so each window receives exactly its patches' global rotary frequencies; this was verified numerically to be bit-identical to slicing the global table per window. The non-spatial `[CLS, register…]` prefix tokens are handled by prepending identity rows (`cos=1, sin=0`) so one position-indexed rotary apply works uniformly in both the folded (local) and merged (global) sequence shapes. The design preserves the full ~`num_windows²`× local-attention FLOP savings of the current backbone, adds only DINOv3's own marginal RoPE cost, exports cleanly to ONNX (no position-embedding monkeypatch needed), and requires no learned positional parameters — but DINOv3 weights are under Meta's proprietary, gated license and **cannot be redistributed under RF-DETR's Apache-2.0**.

---

## 2. The Core Problem: Why DINOv2 Windowing Does Not Transfer to DINOv3

The existing windowed DINOv2 backbone (`dinov2_with_windowed_attn.py`) is **position-agnostic data movement**:

1. The learned absolute position embedding is **added into the token values once**, *before* any windowing, at `Embeddings.forward` line 430: `embeddings = embeddings + self.interpolate_pos_encoding(...)`.
2. Only *after* positions are frozen into the values does the window fold (lines 432–450) reshape/permute the patch grid so each window lives in the batch dimension. Plain dense SDPA inside a layer is then automatically confined to one window.
3. Because position is already baked in, the fold, the global-layer un-fold/merge (`Layer.forward` lines 742–763), and the final feature-map reconstruction (`Backbone.forward` lines 1273–1294) never need to know any coordinate — they are pure index gymnastics.

**DINOv3 deletes the learned position table entirely.** Its *only* positional signal is 2D-axial RoPE applied to query/key vectors *inside every attention call* (`DINOv3ViTAttention.forward`, modeling lines 313–314), based on each token's position *at attention time*. This breaks the DINOv2 assumption in three ways:

- **Position is injected per-layer, not once up front.** You cannot pre-bake positions before folding; coordinates must be threaded into every attention call.
- **RoPE depends on token position at attention time.** After the window-major fold, a token's index in the folded sequence does *not* give its global `(row, col)` — the batch dimension encodes `(image, window_row, window_col)` and tokens within a window are local raster. A naive port that reuses token index would silently misalign frequencies.
- **RoPE is translation-invariant but NOT scale-invariant** (verified: translating coords leaves intra-set scores identical to 6.7e-6; scaling coords by 1.5 changes a score from −1.855 to −0.577). So windowed attention is only correct if every window keeps the **same patch spacing** as the global grid. Renormalizing window coordinates to `[-1,+1]` within the window (step `2/Hpw` instead of `2/Hg`) is wrong — verified score diff 6.39.

The design must therefore (a) compute RoPE from per-token *global* coordinates, (b) make it window-aware, and (c) cleanly handle the `[CLS, register…]` prefix that is non-spatial and gets duplicated `num_windows²` times by the fold.

---

## 3. Chosen Design: Batch-Fold + RoPE-Fold

**Principle:** reuse the proven DINOv2 batch-fold scaffold byte-for-byte; replace only positional injection. Compute the global RoPE table once, then **fold `cos`/`sin` through the identical window-major reshape/permute as the patch tokens** so they physically cannot desync from the tokens they modulate.

Let `B` = batch, `C` = `hidden_size`, `ps` = patch_size (16), `nw` = num_windows, `R` = num_register_tokens (4 for DINOv3), `head_dim = C / num_heads`. Input `H, W` with the constraint `H % (ps*nw) == 0` and `W % (ps*nw) == 0`. Define `Hg = H//ps`, `Wg = W//ps`, `Hpw = Hg//nw`, `Wpw = Wg//nw`, `Tpw = Hpw*Wpw` (patches per window), `P = Hg*Wg` (total patches).

### Step-by-step data flow with exact tensor shapes

| Step | Operation | Output shape |
|---|---|---|
| **0. Patch embed** | `Conv2d(k=s=ps)` → `flatten(2).transpose(1,2)`, h-major raster | `(B, P, C)` |
| **1. (no pos-embed)** | DINOv2 line 430 is **deleted** | `(B, P, C)` |
| **2. Window fold** (DINOv2 lines 438–448) | `view(B, Hg, Wg, C)` → `reshape(B*nw, Hpw, nw, Wpw, C)` → `permute(0,2,1,3,4)` → `reshape(B*nw², Tpw, C)` | `(B*nw², Tpw, C)` |
| **3. Prefix prepend** | `cls.repeat(nw²,1,1)` then `register_tokens.expand(...)` inserted between CLS and patches → per-window order `[CLS, reg₀..reg_{R-1}, patches]` | `(B*nw², R+1+Tpw, C)` |
| **4. RoPE global table** | DINOv3 formula on `(Hg,Wg)` grid, float32 | `cos_g, sin_g : (P, head_dim)` |
| **5. RoPE fold** | fold `cos_g`/`sin_g` with the **same** reshape/permute as Step 2 | `(nw², Tpw, head_dim)` |
| **6. RoPE prefix identity rows** | prepend `R+1` rows of `cos=1, sin=0` | `cos_local, sin_local : (nw², R+1+Tpw, head_dim)` |
| **7. Local block** | SDPA per folded batch element; rotary applied uniformly (identity rows leave prefix unrotated) | `(B*nw², R+1+Tpw, C)` |
| **8. Global block merge** (DINOv2 line 746) | `view(B, nw²*(R+1+Tpw), C)` | `(B, nw²*(R+1+Tpw), C)` |
| **9. RoPE merge** | reshape `cos_local`/`sin_local` with the identical merge → broadcasts to `(B, ...)` | `(1, nw²*(R+1+Tpw), head_dim)` |
| **10. Global attention** | one dense SDPA over merged seq; identity rows keep the `nw²` duplicated prefixes unrotated; patch rows carry true global coords | `(B, nw²*(R+1+Tpw), C)` |
| **11. Re-split** (DINOv2 lines 761–763, reading merged dims) | `view(B*nw², R+1+Tpw, C)`; add folded-shape residual | `(B*nw², R+1+Tpw, C)` |
| **12. Feature map** (DINOv2 lines 1264–1294) | strip `R+1` prefix → inverse fold → `reshape(B,Hg,Wg,C)` → `permute(0,3,1,2)` | `(B, C, Hg, Wg)` |

**Why Step 5/9 are correct (verified numerically, diff = 0.0 exact for the fold; 3e-15 for the shared-table optimization):** folding the global `cos`/`sin` with the same window-major reshape/permute as the patch tokens guarantees window `w`'s row `lidx` equals the global row for that window's patch — i.e. `cos_local[w, lidx] == cos_g[global_idx(w, lidx)]`. In the merged global sequence, the merge applies the same window-major permutation to *both* tokens and `cos`/`sin`, so scores are a consistent permutation of true global-attention scores. RoPE translation-invariance makes this valid: only coordinate *differences* enter the score and every patch keeps its true global spacing.

**Local-block coordinate optimization (grafted from runner-up #3):** because all windows are numerically interchangeable under translation-invariance, the per-window local table can equivalently be built **once** from local indices `0..Hpw-1 / 0..Wpw-1` normalized by the **global** grid size (`coord = 2*(i+0.5)/Hg − 1`, *never* by `Hpw`) and broadcast across all `B*nw²` windows. Verified: this reproduces every window's folded slice to 3e-15. This avoids materializing the full folded local table and is the recommended local path; the explicit fold (Step 5) remains the spec for the **global** block where coordinate-consistency with the merged sequence is the load-bearing property.

**Why identity-row rotary apply (not stock `apply_rotary_pos_emb`):** the stock function (`apply_rotary_pos_emb`, lines 254–266) assumes patches are the *contiguous trailing block* (`num_prefix = num_tokens − num_patches`). For the **local** folded shape `[CLS, reg…, patches]` this trailing-slice assumption actually still holds per window (`num_prefix = R+1`). But for the **global merged** shape `[pfx, win0-patches, pfx, win1-patches, …]` the prefixes are interleaved and patches are not a single trailing block — the trailing-slice apply would silently corrupt positions. Using a single **position-indexed** apply (`q*cos + rotate_half(q)*sin` with per-token `cos`/`sin` including identity prefix rows) works uniformly in both shapes and is the one place a naive reuse breaks.

---

## 4. Code Skeletons

> All new files require the RF-DETR license header:
> ```python
> # ------------------------------------------------------------------------
> # RF-DETR
> # Copyright (c) 2025 Roboflow. All Rights Reserved.
> # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
> # ------------------------------------------------------------------------
> ```

New module: `src/rfdetr/models/backbone/dinov3_with_windowed_attn.py`.

### 4.1 Config class — *newly written* (subclasses HF `DINOv3ViTConfig`)

```python
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig


class WindowedDinov3ViTConfig(DINOv3ViTConfig):
    """DINOv3 ViT config extended with Swin-style windowed-attention fields.

    Inherits all RoPE / architecture fields (rope_theta=100.0, patch_size=16,
    num_register_tokens, use_gated_mlp, key_bias=False, layerscale_value, ...).
    """

    model_type = "windowed_dinov3_vit"

    def __init__(
        self,
        num_windows: int = 1,
        window_block_indexes: list[int] | None = None,
        gradient_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_windows = num_windows
        # local (windowed) layers; layers ABSENT from this list run global attention
        self.window_block_indexes = (
            window_block_indexes
            if window_block_indexes is not None
            else list(range(self.num_hidden_layers))
        )
        self.gradient_checkpointing = gradient_checkpointing
```

### 4.2 Windowed embeddings + RoPE handling — *newly written* (fold copied-with-mods from DINOv2 lines 432–459; pos-embed deleted)

```python
import math
import torch
import torch.nn as nn
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings


def build_rope_tables(num_h, num_w, head_dim, base, device, dtype):
    """DINOv3 2D-axial RoPE on a (num_h, num_w) grid at GLOBAL spacing.

    Returns cos, sin of shape (num_h*num_w, head_dim). Copied-with-mods from
    DINOv3ViTRopePositionEmbedding.forward + get_patches_center_coordinates.
    Compute in float32 to avoid fp16 drift, then cast.
    """
    inv_freq = 1.0 / base ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    ch = torch.arange(0.5, num_h, dtype=torch.float32, device=device) / num_h
    cw = torch.arange(0.5, num_w, dtype=torch.float32, device=device) / num_w
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0                                   # [-1, +1], GLOBAL spacing
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)                         # GPT-NeoX 'half' layout
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def fold_to_windows(t, B, num_h, num_w, nw):
    """Window-major fold IDENTICAL to DINOv2 lines 438-448. Works for patch
    tokens (t: (B, num_h*num_w, C)) AND rope tables (t: (1, num_h*num_w, hd))."""
    Hpw, Wpw = num_h // nw, num_w // nw
    t = t.view(-1, num_h, num_w, t.shape[-1])
    t = t.reshape(t.shape[0] * nw, Hpw, nw, Wpw, t.shape[-1]).permute(0, 2, 1, 3, 4)
    return t.reshape(-1, Hpw * Wpw, t.shape[-1])


class WindowedDinov3ViTEmbeddings(DINOv3ViTEmbeddings):
    """DINOv3 embeddings (cls_token, register_tokens, patch_embeddings Conv2d) +
    window fold. NO learned position table (RoPE replaces it)."""

    def forward(self, pixel_values):
        B, _, H, W = pixel_values.shape
        nw = self.config.num_windows
        divisor = self.config.patch_size * nw
        if H % divisor or W % divisor:
            raise ValueError(f"H,W must be divisible by patch_size*num_windows={divisor}")

        x = self.patch_embeddings(pixel_values.to(self.patch_embeddings.weight.dtype))
        x = x.flatten(2).transpose(1, 2)                          # (B, P, C) h-major
        num_h, num_w = H // self.config.patch_size, W // self.config.patch_size

        if nw > 1:
            x = fold_to_windows(x, B, num_h, num_w, nw)           # (B*nw^2, Tpw, C)

        # prefix [CLS, reg...] per window (DINOv2 lines 449-456 pattern)
        n = x.shape[0]
        cls = self.cls_token.expand(n, -1, -1)
        if self.config.num_register_tokens > 0:
            regs = self.register_tokens.expand(n, -1, -1)
            x = torch.cat([cls, regs, x], dim=1)                  # (B*nw^2, R+1+Tpw, C)
        else:
            x = torch.cat([cls, x], dim=1)
        return x
```

### 4.3 Per-window/merged RoPE table builder — *newly written*

```python
class WindowedRopeEmbedding(nn.Module):
    """Builds BOTH the per-window (local) and merged (global) cos/sin tables,
    each with R+1 identity prefix rows. inv_freq is derived from config; no
    learned params (mirrors DINOv3ViTRopePositionEmbedding non-persistent buffer)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads

    def _identity_prefix(self, ref):
        R1 = 1 + self.config.num_register_tokens
        cos = torch.ones(ref.shape[0], R1, self.head_dim, dtype=ref.dtype, device=ref.device)
        sin = torch.zeros_like(cos)
        return cos, sin

    def forward(self, pixel_values):
        B, _, H, W = pixel_values.shape
        ps, nw = self.config.patch_size, self.config.num_windows
        num_h, num_w = H // ps, W // ps
        cos_g, sin_g = build_rope_tables(num_h, num_w, self.head_dim,
                                         self.config.rope_theta,
                                         pixel_values.device, pixel_values.dtype)
        cos_g, sin_g = cos_g[None], sin_g[None]                   # (1, P, hd)

        # LOCAL: fold to windows, prepend identity prefix -> (nw^2, R+1+Tpw, hd)
        cos_l = fold_to_windows(cos_g, 1, num_h, num_w, nw)
        sin_l = fold_to_windows(sin_g, 1, num_h, num_w, nw)
        pc, ps_ = self._identity_prefix(cos_l)
        cos_l, sin_l = torch.cat([pc, cos_l], 1), torch.cat([ps_, sin_l], 1)

        # GLOBAL: reshape folded+prefixed tables with the SAME merge as tokens
        Tpw_full = cos_l.shape[1]                                 # R+1+Tpw
        cos_m = cos_l.reshape(1, nw * nw * Tpw_full, self.head_dim)
        sin_m = sin_l.reshape(1, nw * nw * Tpw_full, self.head_dim)
        return (cos_l, sin_l), (cos_m, sin_m)
```

### 4.4 Attention forward applying RoPE per window — *copied-with-mods from `DINOv3ViTAttention` (lines 271–334)*; only the rotary apply is replaced

```python
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTAttention, rotate_half, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)


def apply_rope_position_indexed(q, k, cos, sin):
    """Position-indexed rotary apply. cos/sin INCLUDE identity prefix rows, so the
    SAME call works for the folded (local) and merged (global) sequence shapes.
    Replaces stock apply_rotary_pos_emb whose trailing-slice assumption breaks on
    the interleaved-prefix merged sequence."""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class WindowedDinov3ViTAttention(DINOv3ViTAttention):
    def forward(self, hidden_states, position_embeddings, **kwargs):
        cos, sin = position_embeddings                            # (nw^2 or 1, T, hd)
        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # cos/sin broadcast over heads: (n, T, hd) -> (n, 1, T, hd)
        q, k = apply_rope_position_indexed(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        out, _ = attn_fn(self, q, k, v, attention_mask=None,
                         dropout=0.0 if not self.training else self.dropout,
                         scaling=self.scaling, **kwargs)
        out = out.reshape(bsz, seq, -1).contiguous()
        return self.o_proj(out), None
```

### 4.5 Layer with `run_full_attention` merge + consistent cos/sin — *copied-with-mods from DINOv2 `Layer.forward` (lines 734–781) wrapping DINOv3 block (lines 405–450)*

```python
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTLayerScale, DINOv3ViTMLP, DINOv3ViTGatedMLP, DINOv3ViTDropPath,
)


class WindowedDinov3ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_windows = config.num_windows
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = WindowedDinov3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = (DINOv3ViTDropPath(config.drop_path_rate)
                          if config.drop_path_rate > 0 else nn.Identity())
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTGatedMLP(config) if config.use_gated_mlp else DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(self, hidden_states, rope_local, rope_merged, run_full_attention=False):
        shortcut = hidden_states                                  # folded-shape residual
        x = self.norm1(hidden_states)
        if run_full_attention:                                    # DINOv2 lines 742-748
            bw, tpw, c = x.shape
            nw2 = self.num_windows ** 2
            x = x.view(bw // nw2, nw2 * tpw, c)
            pos = rope_merged                                     # (1, nw^2*T, hd)
        else:
            pos = rope_local                                      # (nw^2, T, hd)

        attn_out, _ = self.attention(x, position_embeddings=pos)

        if run_full_attention:                                    # DINOv2 lines 758-763
            bw, tpw, c = x.shape                                  # ALREADY merged dims
            nw2 = self.num_windows ** 2
            attn_out = attn_out.view(bw * nw2, tpw // nw2, c)

        hidden_states = self.drop_path(self.layer_scale1(attn_out)) + shortcut
        residual = hidden_states
        out = self.mlp(self.norm2(hidden_states))
        return self.drop_path(self.layer_scale2(out)) + residual
```

### 4.6 Encoder local-vs-global routing — *copied-with-mods from DINOv2 `Encoder.forward` (lines 791–835)*

```python
class WindowedDinov3ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [WindowedDinov3ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, rope_local, rope_merged):
        all_hidden = ()
        last_feature_idx = int(self.config._out_features[-1][5:])  # 'stageN' -> N
        for i, layer in enumerate(self.layer):
            all_hidden = all_hidden + (hidden_states,)
            if i > last_feature_idx:
                break
            run_full = i not in self.config.window_block_indexes
            hidden_states = layer(hidden_states, rope_local, rope_merged, run_full)
        all_hidden = all_hidden + (hidden_states,)
        return all_hidden
```

### 4.7 Backbone feature extraction — *copied-with-mods from DINOv2 `Backbone.forward` (lines 1256–1296)*

```python
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTPreTrainedModel


class WindowedDinov3ViTBackbone(DINOv3ViTPreTrainedModel):
    config_class = WindowedDinov3ViTConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = WindowedDinov3ViTEmbeddings(config)
        self.rope = WindowedRopeEmbedding(config)
        self.encoder = WindowedDinov3ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_register_tokens = config.num_register_tokens
        self.stage_names = config.stage_names
        self.out_features = config._out_features
        self.post_init()

    def forward(self, pixel_values):
        B, _, H, W = pixel_values.shape
        nw, ps = self.config.num_windows, self.config.patch_size
        num_h, num_w = H // ps, W // ps
        x = self.embeddings(pixel_values)
        rope_local, rope_merged = self.rope(pixel_values)
        hidden_states = self.encoder(x, rope_local, rope_merged)

        feature_maps = []
        for stage, hs in zip(self.stage_names, hidden_states):
            if stage not in self.out_features:
                continue
            hs = self.layernorm(hs)
            hs = hs[:, self.num_register_tokens + 1:]             # strip prefix
            if nw > 1:                                            # inverse fold (DINOv2 1273-1291)
                nw2 = nw * nw
                bw, tpw, c = hs.shape
                Hpw, Wpw = num_h // nw, num_w // nw
                hs = hs.reshape(bw // nw2, nw2 * tpw, c)
                hs = hs.reshape((bw // nw2) * nw, nw, Hpw, Wpw, c).permute(0, 2, 1, 3, 4)
            hs = hs.reshape(B, num_h, num_w, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(hs)
        return feature_maps
```

**Reuse summary:** Steps 2/8/11/12 (fold, merge, re-split, inverse fold) and the encoder routing are copied **verbatim** from the DINOv2 scaffold; the per-layer block, attention projections, MLP/GatedMLP, layer-scale, and RoPE formula are composed from **stock `transformers` DINOv3ViT submodules**. Newly written: the windowed embeddings (pos-embed deleted), the RoPE table builder (`build_rope_tables` + fold + identity prefix), and the position-indexed rotary apply.

---

## 5. Weight Loading

### What loads, and how

- **RoPE has zero state-dict keys.** `inv_freq` is a non-persistent buffer (`register_buffer(..., persistent=False)`, modeling line 166), re-derived from `rope_theta` and `head_dim`. There is **no learned absolute position table anywhere** in DINOv3 — the only positional signal is RoPE.
- Because the per-layer block is **composed from stock DINOv3ViT submodules**, checkpoint keys map ~1:1. The loadable parameters from `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m` are:
  - `embeddings.cls_token`, `embeddings.register_tokens` (4), `embeddings.patch_embeddings.{weight,bias}` (Conv2d k=s=16),
  - per layer `norm1/norm2`, `layer_scale1/2.lambda1`, `attention.{q,v,o}_proj.{weight,bias}` and `attention.k_proj.weight` (**no `k_proj.bias`** — `key_bias=False`, config line 87),
  - `mlp.{up_proj,down_proj}` (plain, ViT-S/B/L) **or** `mlp.{gate_proj,up_proj,down_proj}` (SwiGLU, ViT-S+/H+) selected by `use_gated_mlp`,
  - final `layernorm` / `norm`.
- **Only remapping needed** is the top-level module-prefix substitution between RF-DETR's namespace (`encoder.embeddings.*`, `encoder.encoder.layer.{i}.*`, `encoder.layernorm`) and HF's (`embeddings.*`, `model.layer.{i}.*`, `norm.*`). No per-tensor reshaping — windowing is pure runtime reshape and changes no weight shapes.
- **A symmetric q/k/v-bias assumption will fail** to load (`k_proj` has no bias tensor). Branch on FFN type for S+/H+ SwiGLU.

### Constraints

- **`patch_size = 16` is fixed** for all DINOv3 ViT checkpoints. RF-DETR's current configs already use `patch_size=16` (e.g. `RFDETRMediumConfig` line 457, `RFDETRLargeConfig` line 472), which already disables DINOv2 hub-weight loading. So in practice **DINOv3 RF-DETR variants load from RF-DETR finetuned checkpoints**, not raw hub weights (see §6 guards).
- **Divisibility:** `H` and `W` must each be divisible by `patch_size * num_windows = 16 * nw`. This is enforced in both `Embeddings.forward` (ValueError) and `DinoV2.forward` (assert). With `nw=4`, resolution must be a multiple of 64; with `nw=2`, a multiple of 32. The existing `RFDETRMediumConfig` uses `resolution=576, num_windows=2` (576 % 32 = 0 ✓); `RFDETRLargeConfig` uses `resolution=704, num_windows=2` (704 % 32 = 0 ✓). Any new DINOv3 variant must keep `resolution % (16*nw) == 0`.

---

## 6. Integration Edits (file → symbol → change)

| File | Symbol | Change |
|---|---|---|
| `src/rfdetr/config.py` (line 16) | `EncoderName` Literal | **Widen** to add `"dinov3_windowed_small"`, `"dinov3_windowed_base"`, `"dinov3_windowed_large"` (and `"dinov3_registers_windowed_*"` if a no-register branch is wanted). |
| `src/rfdetr/config.py` (~line 395+) | New `RFDETR*Config` classes | **Add** config classes with `encoder="dinov3_windowed_*"`, `patch_size=16`, `num_windows`, `resolution % (16*nw)==0`, `positional_encoding_size = resolution//16`, `out_feature_indexes` re-picked per depth, matching `pretrain_weights`. |
| `src/rfdetr/config.py` (lines 264–281) | `breaking_fields` (`"encoder"`) | **Review:** ensure each new variant's `encoder` default matches so the compat warning is accurate. |
| `src/rfdetr/config.py` (lines 155–189) | `_sync_pe_with_resolution` | **Review (no crash):** `positional_encoding_size` is semantically dead for a RoPE backbone (RoPE derives positions at runtime). Keep as the integer grid size for divisibility bookkeeping. |
| `src/rfdetr/models/backbone/backbone.py` (line 62) | `assert name_parts[0] == "dinov2"` | **Relax:** accept `"dinov2"` or `"dinov3"`; detect family and pass a `family`/`use_dinov3` flag to `DinoV2(...)` (or a parallel wrapper). |
| `src/rfdetr/models/backbone/backbone.py` (lines 75–87) | `DinoV2(...)` instantiation | **Add** family argument so the right hub id / config / backbone class is selected. |
| `src/rfdetr/models/backbone/dinov2.py` (lines 25–30) | `size_to_width` | **Add** DINOv3 entries: `small=384` (ViT-S), `base=768` (ViT-B), `large=1024` (ViT-L). `tiny=192` has **no** DINOv3 match. Disambiguate ViT-S vs ViT-S+ by `use_gated_mlp`, **not** width. |
| `src/rfdetr/models/backbone/dinov2.py` (lines 32–52) | `size_to_config` / `get_config` | **Add** a `dinov3_configs/` dir with `dinov3_{small,base,large}.json` (RoPE fields: `rope_theta=100.0`, `patch_size=16`, `num_register_tokens=4`, `layerscale_value=1.0`, `layer_norm_eps=1e-5`, `use_gated_mlp` per variant). |
| `src/rfdetr/models/backbone/dinov2.py` (line 72) | hardcoded `facebook/dinov2-*` | **Branch** to `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m`. |
| `src/rfdetr/models/backbone/dinov2.py` (lines 93–146) | windowed instantiation | **Branch** on family to instantiate `WindowedDinov3ViTConfig` / `WindowedDinov3ViTBackbone`. |
| `src/rfdetr/models/backbone/dinov2.py` (lines 151–208) | `DinoV2.export()` | **Guard:** wraps in `if hasattr(self.encoder.embeddings, "position_embeddings")`. A RoPE backbone has no such attr → would `AttributeError` (reads line 190, reassigns line 205, monkeypatches `interpolate_pos_encoding`). Skip the whole monkeypatch for DINOv3. |
| `src/rfdetr/models/weights.py` (lines 35, 199–255, 471) | `_PE_KEY_SUFFIX` / `interpolate_position_embeddings` | **Review (harmless no-op):** finds zero matching keys for a RoPE checkpoint → safe. **Verify** no *other* resolution-dependent learnable param exists (none — RoPE is dynamic), else `load_state_dict` shape mismatch. |
| `src/rfdetr/models/lwdetr.py` (lines 389–428) | `build_model` | **No change:** routing forwards `encoder`/`patch_size`/`num_windows`/`positional_encoding_size` generically. |
| `src/rfdetr/_namespace.py` (lines 29, 45, 151) | `_MC_NAMESPACE_FIELDS` | **No change:** `encoder` and `positional_encoding_size` copied verbatim. |
| `src/rfdetr/models/position_encoding.py` (lines 138–148) | `build_position_encoding` | **No change:** this is the decoder's sine PE over projector features — formula-derived, independent of backbone RoPE. Does not break. |
| `src/rfdetr/detr.py` (lines 586–600) | `train()` PE update | **Review (no crash):** only manipulates integer `positional_encoding_size`; semantics change for RoPE but no failure. |

---

## 7. Recommended Checkpoint per RF-DETR Tier

| RF-DETR tier | `size_to_width` | DINOv3 checkpoint | width / depth / heads | FFN | Notes |
|---|---|---|---|---|---|
| tiny | 192 | **none** | — | — | DINOv3's smallest ViT is width 384; use ConvNeXt-tiny or distill/truncate. |
| small | 384 | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 / 12 / 6 | plain MLP | ViT-S+ (`dinov3-vits16plus`) is also width 384 but SwiGLU — disambiguate by `use_gated_mlp`. |
| base | 768 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 / 12 / 12 | plain MLP | |
| large | 1024 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 / **24** / 16 | plain MLP, `intermediate_size=4096` | **24 layers** — re-pick `out_feature_indexes` (DINOv2 default `[2,4,5,9]` assumes 12 layers); `window_block_indexes` derives from it. |

All DINOv3 ViTs use `patch_size=16`, `num_register_tokens=4`, `rope_theta=100.0`, `layerscale_value=1.0`, `image_size=224`.

**patch_size 14 → 16 consequence:** DINOv2 used patch 14; DINOv3 uses 16. At a fixed resolution this yields **fewer patches** (640/16 = 40 vs 640/14 ≈ 45), so attention is modestly cheaper. The downside: resolution choices must satisfy `% (16*nw) == 0` (multiple of 64 for `nw=4`, 32 for `nw=2`), which differs from the DINOv2 `% (14*nw)` lattice. Existing RF-DETR `patch_size=16` configs already satisfy this.

---

## 8. Latency Analysis vs Current DINOv2 Windowed Backbone

**FLOP savings are fully preserved** — the window fold is byte-for-byte the same reshape/permute data movement, so local layers still run SDPA on `(B*nw², R+1+Tpw, head_dim)`:

- Dominant patch-attention term drops from global `O((Hg·Wg)²)` to `O(nw² · (Hg·Wg/nw²)²) = O((Hg·Wg)²/nw²)` — the same ~`nw²`× reduction. E.g. 640px / patch16: global = 40×40 = 1600 patches; `nw=4` → 16 windows of 100 patches → ~16× fewer attention FLOPs on local layers.
- **Added cost vs DINOv2:** (a) one RoPE `cos`/`sin` computation per forward (small elementwise ops on `(P, head_dim)`, float32, `lru_cache`-able on `(num_h, num_w)`); (b) the rotary apply (two elementwise mul+add on q,k) per attention — exactly the marginal cost DINOv3 itself pays, *replacing* DINOv2's pos-embed-add. The shared-local-table optimization makes local-block RoPE setup `O(Tpw·head_dim)` once.
- **Global output layers** cost the same full-image attention as DINOv2's merge, including the `nw²`-duplicated prefixes (merged length `nw²·(R+1+Tpw)`). Optional prefix-dedup global block (grafted from runner-up #3, gated behind a flag) removes `(nw²−1)·(R+1)` redundant prefix tokens for a small global-block win.
- **patch_size 16 vs 14** yields fewer patches → modestly faster.

**Net:** latency parity with the DINOv2 windowed backbone, plus negligible RoPE overhead. (For contrast, the rejected masked/flex-attention approach measured ~14–40× more attention FLOPs with dense masks and 5–10× more with flex BlockMask at 128-block granularity vs 81–196-token windows — it fails the latency gate and flex is not ONNX-exportable.)

---

## 9. Licensing Caveat

**This is a hard blocker for redistribution.** DINOv2 is Apache-2.0 (RF-DETR currently bundles it freely). **DINOv3 is under Meta's proprietary "DINOv3 License":** gated on HuggingFace (requires accepting terms + sharing contact info; unauthenticated `config.json` access returns HTTP 401), permits commercial use but **requires the DINOv3 Agreement to accompany every redistribution** of weights/derivatives and **forbids relicensing/sublicensing under Apache-2.0**, and carries an Acceptable Use Policy.

**Implication:** RF-DETR may ship **Apache-2.0 *code*** that loads a user's own gated HF download, but **must NOT bundle, host, or present DINOv3 weights as Apache-2.0**. CI/automated downloads need an HF token from a user who accepted the license. The `load_dinov2_weights` flag's "load from `facebook/dinov2-*`" semantics must be re-pointed to the gated DINOv3 repo, not merely toggled.

---

## 10. Test Plan & Phased Roadmap

### Unit tests (`tests/`, CPU, `-m "not gpu"`)
1. **Divisibility:** `WindowedDinov3ViTBackbone.forward` raises `ValueError` when `H % (16*nw) != 0`; parametrize valid/invalid shapes.
2. **Shape:** for `(B, 3, H, W)` input and `out_feature_indexes`, each returned feature map is `(B, size_to_width[size], H//16, W//16)`.
3. **Weight load:** construct from a `WindowedDinov3ViTConfig` and assert a stock `DINOv3ViTBackbone` state_dict (after prefix remap) loads with zero unexpected/missing keys except the expected RoPE buffer absence; assert `k_proj.bias` is absent.
4. **RoPE-fold identity:** assert folded `cos_local[w, lidx]` equals global `cos_g[global_idx(w, lidx)]` for all windows (the verified diff=0.0 property).
5. **Equivalence at `num_windows=1`:** with `nw=1` and all layers global, assert the windowed module's output matches a stock `transformers` `DINOv3ViTBackbone` to fp32 tolerance — proves the port reduces exactly to upstream DINOv3.
6. **Equivalence oracle (test-only):** implement the dense additive-mask raster-order path (eager SDPA, no compile, no flex) as a numerical oracle; assert the batch-fold windowed output matches the masked-global output for intra-window pairs, catching window-major RoPE misalignment. Use the explicit window-id predicate `window_id = (r//Hpw)*nw + (c//Wpw)` (prefix = −1) as the authoritative membership spec.
7. **Prefix unrotated:** assert identity-row apply leaves `[CLS, reg…]` tokens bit-unchanged.

### Integration tests (mark `@pytest.mark.gpu` for heavy paths)
8. **End-to-end forward:** build an `RFDETR*` model with a `dinov3_windowed_*` encoder; run a forward on a dummy image; assert detection outputs have correct shapes.
9. **Tiny train step:** one optimizer step on a 1–2 image batch; assert loss is finite and parameters update (RoPE buffers do not).
10. **ONNX export:** export at a fixed resolution; assert traced output matches eager within tolerance and the export does **not** touch `position_embeddings` (guard exercised).

### Phased roadmap
- **Phase 1 — Prototype:** new module + config; load `facebook/dinov3-vits16` weights (with HF token); pass unit tests 1–7 with `load` from hub.
- **Phase 2 — Validate:** wire routing (config Literal, backbone parser, `dinov2.py` branch, export/weights guards); pass integration tests 8–10; confirm latency parity with DINOv2 windowed via a microbenchmark.
- **Phase 3 — Retrain:** train RF-DETR detection heads on the DINOv3 windowed backbone per tier; produce RF-DETR finetuned checkpoints (the redistributable artifact — code is Apache-2.0, the RF-DETR-trained weights are RF-DETR's to license, but the **DINOv3-derived** initialization carries the DINOv3 License — confirm with legal before publishing any checkpoint that embeds DINOv3-pretrained weights).

> **Open a maintainer issue before implementation** (per AGENTS.md): adding a new backbone family requires approval on approach.

---

## 11. Honest Risks & Open Questions

**Risks**
- **Global-merge RoPE consistency** is the single most error-prone surface: the merged `cos`/`sin` must be reshaped with the *identical* merge as the tokens, and the duplicated-prefix identity rows must line up. Mitigated by the equivalence oracle (test 6).
- **Must not reuse stock `apply_rotary_pos_emb`** in the merged shape — its trailing-slice prefix assumption silently corrupts positions when prefixes are interleaved. The custom position-indexed apply must be tested in both shapes.
- **Non-square inputs** inherit the DINOv2 copied-bug H/W reshape-order caveat (comment at DINOv2 lines 1265–1266). The RoPE fold uses the same ordering so it stays *consistent*, but non-square support needs explicit testing.
- **`out_feature_indexes` re-pick** for ViT-L (24 layers) is required and load-bearing — `window_block_indexes` derives from it and a wrong pick changes which layers are global.
- **DINOv3 prefix duplication semantics** (the `nw²` CLS/register copies in global attention) are faithful to the existing DINOv2 contract but differ from vanilla DINOv3. Acceptable, but document it; the optional prefix-dedup global block is the cleaner-semantics alternative.
- **License/gating** is a non-engineering blocker on shipping weights and on CI automation.

**Open questions**
1. **Intended local-coordinate semantics:** the design uses *global*-spacing coordinates for local windows (matching how DINOv2's pre-fold abs-PE encoded full-image position). Confirm this is the desired DINOv3 behavior vs per-window-reset coordinates — the latter would change semantics and is numerically wrong for cross-window-equivalence (verified).
2. **Prefix-dedup global block:** ship it (cleaner single-CLS semantics, small latency win) or keep verbatim-DINOv2 merge (bit-compatible with the existing windowing contract)? Recommend default verbatim, flag for dedup.
3. **ViT-S vs ViT-S+ for the `small` tier:** plain MLP (S) or SwiGLU (S+)? Both width 384; choose by accuracy/latency tradeoff, set `use_gated_mlp` accordingly, and verify the exact repo id (`dinov3-vits16plus` vs `dinov3-vitsplus` naming inconsistency in docs).
4. **`pretrain_weights` story:** since `patch_size=16` already disables hub-weight loading in the DINOv2 path, do DINOv3 variants always start from RF-DETR finetuned checkpoints, or is there a one-time "init from gated DINOv3 hub" bootstrap path that needs an HF token in CI?
5. **Tiny tier:** drop it for DINOv3, or substitute a ConvNeXt-tiny (no windowing needed) — but ConvNeXt is under the same gated license.
6. **`positional_encoding_size` semantics:** keep as the integer grid size for divisibility bookkeeping, or introduce a sentinel for RoPE backbones to make the dead semantics explicit?