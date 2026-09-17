#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VGGT-Det style attention-map visualization.

This script reproduces the attention map defined in the official VGGT-Det
repository (yangcaoai/VGGT-Det-CVPR2026, vggt/models/aggregator.py +
vggt/layers/attention.py), NOT the query-box variant in visualize_attention.py.

VGGT-Det's attention map is a per-patch "saliency" map computed as:

    1. take one attention layer (by default the FIRST frame-wise block, f0,
       see `vis_layer_idx = 0` and `vis_attn_type = "frame"` in aggregator.py),
    2. materialize the softmax attention matrix A = softmax(Q K^T / sqrt(D)),
    3. average over heads, then average over all QUERY rows (column mean), i.e.
           s[n] = mean_{head, row} A[row, n],
       giving, for every key token, how much attention it receives on average,
    4. drop the camera + register tokens (keep patch tokens only),
    5. reshape to the patch grid (gh, gw).

There is NO query box involved: the map is per-frame and independent of any
box. (VGGT-Det additionally zeroes low-depth regions with a depth mask and
applies per-image min-max normalization downstream for point sampling; the
depth-mask step is omitted here because it requires the depth head.)

Examples
--------
    # VGGT-Det default: frame block 0, all frames, min-max per frame
    python visualize_attention_vggtdet.py --images examples/room/images --out room_vggtdet.png

    # only the first frame, two-panel view
    python visualize_attention_vggtdet.py --images examples/room/images --frame 0

    # use a different layer (frame block 12) and percentile normalization
    python visualize_attention_vggtdet.py --images examples/kitchen/images \
        --layer f12 --norm percentile --save-npz kitchen_vggtdet.npz
"""

import argparse
import os
import sys

# Windows conda setups often carry two OpenMP runtimes (torch + MKL); without this
# the process can abort with "OMP: Error #15" before any of our code runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Image loading / preprocessing (mirrors vggt.utils.load_fn.load_and_preprocess_images)
# ---------------------------------------------------------------------------

def collect_image_paths(inputs):
    """Expand folders / file lists into a sorted list of image paths."""
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(
                str(p) for p in sorted(Path(item).iterdir()) if p.suffix.lower() in IMAGE_EXTS
            )
        elif os.path.isfile(item):
            paths.append(item)
        else:
            raise FileNotFoundError(f"No such file or directory: {item}")
    if not paths:
        raise ValueError("No images found in the given paths")
    return paths


def preprocess_images(paths, mode="crop", target_size=518, patch_size=14):
    """
    Load images exactly like vggt.utils.load_fn.load_and_preprocess_images
    ("crop" or "pad" mode) but also return, per image, the original->processed
    transform needed to map heatmaps between the two coordinate systems.

    Returns:
        images (np.ndarray): (S, H, W, 3) float32 in [0, 1]
        transforms (list[dict]): per-image transform with keys
            scale_x, scale_y   processed = original * scale + offset
            off_x, off_y
            orig_w, orig_h     original image size
            proc_w, proc_h     processed size (after padding to a common size)
    """
    arrs, transforms = [], []
    for path in paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
        img = img.convert("RGB")
        w, h = img.size

        if mode == "crop":
            new_w = target_size
            new_h = round(h * (new_w / w) / patch_size) * patch_size
            im2 = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            top = (new_h - target_size) // 2 if new_h > target_size else 0
            if new_h > target_size:
                im2 = im2.crop((0, top, new_w, top + target_size))
            t = dict(scale_x=target_size / w, scale_y=new_h / h, off_x=0.0, off_y=-float(top))
        else:  # pad
            if w >= h:
                new_w = target_size
                new_h = round(h * (new_w / w) / patch_size) * patch_size
            else:
                new_h = target_size
                new_w = round(w * (new_h / h) / patch_size) * patch_size
            im2 = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            left = (target_size - new_w) // 2
            top = (target_size - new_h) // 2
            canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            canvas.paste(im2, (left, top))
            im2 = canvas
            t = dict(scale_x=new_w / w, scale_y=new_h / h, off_x=float(left), off_y=float(top))

        t.update(orig_w=w, orig_h=h)
        arrs.append(np.asarray(im2, dtype=np.float32) / 255.0)
        transforms.append(t)

    # pad all images to a common size (load_fn pads white, centered), adjust offsets
    proc_h = max(a.shape[0] for a in arrs)
    proc_w = max(a.shape[1] for a in arrs)
    for i, arr in enumerate(arrs):
        pad_h = proc_h - arr.shape[0]
        pad_w = proc_w - arr.shape[1]
        if pad_h or pad_w:
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            arrs[i] = np.pad(arr, ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)), constant_values=1.0)
            transforms[i]["off_x"] += pad_left
            transforms[i]["off_y"] += pad_top
        transforms[i]["proc_w"] = proc_w
        transforms[i]["proc_h"] = proc_h

    return np.stack(arrs), transforms


# ---------------------------------------------------------------------------
# Attention capture (VGGT-Det saliency map)
# ---------------------------------------------------------------------------

class VGGTDetAttentionCapture:
    """
    Captures VGGT-Det's per-patch attention saliency from a single layer.

    For the selected layer, the full softmax attention matrix is never
    materialized at once; instead the column-sum over all query rows and heads
    is accumulated in float32 in chunks:

        acc[b, n] = sum_{head, row} softmax(Q K^T / sqrt(D))[row, n]

    which is exactly what VGGT-Det computes as
        attn.mean(heads).mean(query_rows)   (up to the final 1/(H*N) scale).
    """

    ROW_CHUNK = 256  # query rows per fp32 matmul chunk

    def __init__(self, aggregator, layer_tag, num_frames, P, patch_start_idx):
        self.aggregator = aggregator
        self.kind, self.layer_idx = layer_tag  # 'frame' | 'global', int index
        self.num_frames = int(num_frames)
        self.P = int(P)
        self.patch_start_idx = int(patch_start_idx)

        self._current = None
        self._installed = []      # (attn_module, orig_forward)
        self._orig_sdpa = None
        self._acc = None          # (B, N) float32 on CPU, column-sum over heads+rows
        self._n_heads = None
        self._seq = None

    def __enter__(self):
        self._install_hook()
        self._patch_sdpa()
        return self

    def __exit__(self, *exc):
        self._restore()

    # -- installation -------------------------------------------------------

    def _install_hook(self):
        if self.kind == "frame":
            blocks = self.aggregator.frame_blocks
        else:
            blocks = self.aggregator.global_blocks
        attn = blocks[self.layer_idx].attn
        orig_forward = attn.forward
        capture = self

        def tagged_forward(x, pos=None, _orig=orig_forward):
            prev, capture._current = capture._current, True
            try:
                return _orig(x, pos=pos)
            finally:
                capture._current = prev

        attn.forward = tagged_forward
        self._installed.append((attn, orig_forward))

    def _patch_sdpa(self):
        capture = self
        capture._orig_sdpa = F.scaled_dot_product_attention

        def sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if capture._current is not None and query.dim() == 4:
                capture._capture(query, key)
            return capture._orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
            )

        F.scaled_dot_product_attention = sdpa_wrapper

    def _restore(self):
        for attn, _orig in self._installed:
            attn.__dict__.pop("forward", None)
        self._installed = []
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None
        self._current = None

    # -- capture ------------------------------------------------------------

    @torch.no_grad()
    def _capture(self, q, k):
        B, H, N, D = q.shape
        # frame blocks run on (B*S, P, C): one batch element per frame (scene batch 1)
        # global blocks run on (B, S*P, C) with scene batch size 1.
        if self.kind == "frame":
            if B != self.num_frames:
                raise RuntimeError(
                    f"Attention visualization requires scene batch size 1 (got batch {B}, expected {self.num_frames})"
                )
            expected_N = self.P
        else:
            if B != 1:
                raise RuntimeError(
                    f"Attention visualization requires scene batch size 1 (got batch {B}, expected 1)"
                )
            expected_N = self.num_frames * self.P

        if N != expected_N:
            raise RuntimeError(f"Sequence length {N} != expected {expected_N}")

        k32 = k.float()
        acc = torch.zeros(B, N, device=q.device, dtype=torch.float32)
        for s in range(0, N, self.ROW_CHUNK):
            r = slice(s, min(s + self.ROW_CHUNK, N))
            q_sel = q[:, :, r, :].float()                       # (B, H, chunk, D)
            logits = torch.matmul(q_sel, k32.transpose(-2, -1)) * (D ** -0.5)
            attn = logits.softmax(dim=-1).sum(dim=1)            # (B, chunk, N), sum over heads
            acc += attn.sum(dim=1)                              # (B, N), sum over query rows
        self._acc = acc.cpu()
        self._n_heads = int(H)
        self._seq = int(N)


def saliency_maps(capture, num_frames, P, patch_start_idx, gw, gh):
    """
    Convert the captured column-sum into per-frame (gh, gw) saliency maps.

    VGGT-Det: attn.mean(heads).mean(query_rows) = acc / (H * N).
    """
    acc = capture._acc                          # (B, N)
    mean = acc / (capture._n_heads * capture._seq)   # column mean over all query rows & heads
    if capture.kind == "frame":
        # (S, P) -> keep patch tokens only
        vec = mean[:, patch_start_idx:]
    else:
        # (1, S*P) -> (S, P) -> keep patch tokens only
        vec = mean.view(num_frames, P)[:, patch_start_idx:]
    npatch = gw * gh
    if vec.shape[1] != npatch:
        raise RuntimeError(f"Expected {npatch} patch tokens, got {vec.shape[1]}")
    return vec.reshape(num_frames, gh, gw).numpy()


# ---------------------------------------------------------------------------
# Map normalization / rendering
# ---------------------------------------------------------------------------

def normalize_maps(maps, norm="minmax", percentiles=(2.0, 98.0)):
    """Per-frame normalization: min-max (VGGT-Det) or percentile clipping."""
    S = maps.shape[0]
    out = np.empty_like(maps)
    for i in range(S):
        m = maps[i]
        if norm == "minmax":
            lo, hi = m.min(), m.max()
            out[i] = (m - lo) / (hi - lo + 1e-12) if hi - lo > 1e-12 else np.zeros_like(m)
        else:  # percentile
            lo, hi = np.percentile(m, percentiles)
            if hi - lo < 1e-12:
                out[i] = np.zeros_like(m)
            else:
                out[i] = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    return out


def _overlay_extent(t):
    """Map the processed image rectangle back to original pixel coordinates."""
    x0 = (0.0 - t["off_x"]) / t["scale_x"]
    x1 = (t["proc_w"] - t["off_x"]) / t["scale_x"]
    y0 = (0.0 - t["off_y"]) / t["scale_y"]
    y1 = (t["proc_h"] - t["off_y"]) / t["scale_y"]
    return x0, x1, y0, y1


def render_single(orig_rgb, heatmap, t, out_path, cmap="jet", alpha=0.55,
                  title_left="Input Image", title_right="Attention (VGGT-Det)"):
    """Two-panel figure for a single frame."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    ax_l.imshow(orig_rgb)
    ax_l.set_title(title_left, fontsize=15)
    ax_l.axis("off")

    ax_r.imshow(orig_rgb)
    ax_r.set_title(title_right, fontsize=15)
    ax_r.axis("off")
    x0, x1, y0, y1 = _overlay_extent(t)
    ax_r.imshow(
        heatmap, cmap=cmap, alpha=alpha, interpolation="bilinear",
        origin="upper", extent=(x0, x1, y1, y0),
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_grid(orig_rgbs, heatmaps, transforms, out_path, cmap="jet", alpha=0.55):
    """Grid figure: one column per frame, top = original, bottom = overlay."""
    S = len(orig_rgbs)
    fig, axes = plt.subplots(2, S, figsize=(4 * S, 9), squeeze=False, constrained_layout=True)
    for i in range(S):
        axes[0, i].imshow(orig_rgbs[i])
        axes[0, i].set_title(f"Frame {i}", fontsize=13)
        axes[0, i].axis("off")

        axes[1, i].imshow(orig_rgbs[i])
        axes[1, i].set_title(f"Frame {i} attention", fontsize=13)
        axes[1, i].axis("off")
        x0, x1, y0, y1 = _overlay_extent(transforms[i])
        axes[1, i].imshow(
            heatmaps[i], cmap=cmap, alpha=alpha, interpolation="bilinear",
            origin="upper", extent=(x0, x1, y1, y0),
        )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights=None, device="cuda", dtype=torch.bfloat16):
    """
    Build the VGGT model and load its weights.

    The Hugging Face path deliberately loads the safetensors file (mmap) BEFORE
    constructing the model and assigns parameters with load_state_dict(assign=True):
    on memory-tight Windows machines (e.g. 16 GB RAM), constructing the 4.8 GB
    fp32 model first and then mapping the 4.7 GB weight file can exhaust the
    commit limit and crash the process, while this order never materializes a
    full second copy of the weights.
    """
    from vggt.models.vggt import VGGT

    if weights:
        print(f"Loading VGGT weights from {weights}")
        state_dict = torch.load(weights, map_location="cpu")
        model = VGGT()
        model.load_state_dict(state_dict)
        del state_dict
    else:
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st

        path = hf_hub_download("facebook/VGGT-1B", "model.safetensors")
        print(f"Loading VGGT weights from {path}")
        state_dict = st.load_file(path)  # mmap-backed, no full read into RAM
        model = VGGT()
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_layer(spec, depth):
    """Parse a single layer tag like 'f0' or 'g23' (f=frame-wise, g=global)."""
    spec = spec.strip().lower()
    if len(spec) >= 2 and spec[0] in ("f", "g") and spec[1:].isdigit():
        idx = int(spec[1:])
        if not 0 <= idx < depth:
            raise ValueError(f"Layer index {idx} out of range [0, {depth - 1}]")
        return ("frame" if spec[0] == "f" else "global", idx)
    raise ValueError(f"Cannot parse layer '{spec}' (expected e.g. f0, f12, g23)")


def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT-Det style attention-map visualization (per-patch saliency, no query box)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", nargs="+", required=True,
                   help="Image files and/or folders (frames of one scene, sorted by name)")
    p.add_argument("--layer", default="f0",
                   help="Single attention layer to use, e.g. 'f0' (VGGT-Det default), 'f12', 'g23' "
                        "(f=frame-wise, g=global block, 0..23)")
    p.add_argument("--frame", type=int, default=None,
                   help="Render only this frame index (default: render all frames in a grid)")
    p.add_argument("--mode", choices=("crop", "pad"), default="crop",
                   help="Preprocessing mode, same as load_and_preprocess_images")
    p.add_argument("--weights", default=None,
                   help="Path to a local VGGT checkpoint (state dict). Default: download facebook/VGGT-1B")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    p.add_argument("--norm", choices=("minmax", "percentile"), default="minmax",
                   help="Per-frame normalization: min-max (VGGT-Det) or percentile clipping")
    p.add_argument("--percentile", nargs=2, type=float, default=(2.0, 98.0), metavar=("LO", "HI"),
                   help="Percentile clip used when --norm percentile")
    p.add_argument("--cmap", default="jet", help="Matplotlib colormap for the heatmap")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity")
    p.add_argument("--out", default="attention_vggtdet.png", help="Output figure path")
    p.add_argument("--save-npz", default=None, help="Optionally save the raw per-frame maps (.npz)")
    p.add_argument("--list-layers", action="store_true", help="Print available layer tags and exit")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"auto": None, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    paths = collect_image_paths(args.images)
    print(f"Using {len(paths)} frames")
    if args.list_layers:
        print("VGGT-Det default: depth=24 -> f0..f23 (frame-wise blocks), g0..g23 (global blocks)")
        print("VGGT-Det uses the first frame-wise block: --layer f0")
        return

    images_np, transforms = preprocess_images(paths, mode=args.mode)
    S, H, W, _ = images_np.shape
    patch_size = 14

    model = load_model(args.weights, device, dtype)
    agg = model.aggregator

    depth = len(agg.frame_blocks)
    layer_tag = parse_layer(args.layer, depth)
    print(f"Layer used: {layer_tag[0]}{layer_tag[1]}")

    patch_start_idx = int(agg.patch_start_idx)          # 1 + num_register_tokens (=5 for VGGT-1B)
    gw, gh = W // patch_size, H // patch_size
    P = patch_start_idx + gw * gh
    print(f"Grid {gw}x{gh} patches, P={P} tokens/frame, S={S} frames")

    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)  # (1, S, 3, H, W)

    capture = VGGTDetAttentionCapture(
        agg, layer_tag, num_frames=S, P=P, patch_start_idx=patch_start_idx,
    )
    print("Running aggregator forward with attention capture ...")
    with torch.no_grad(), capture:
        agg(images)

    if capture._acc is None:
        raise RuntimeError(
            f"No attention captured for layer {layer_tag[0]}{layer_tag[1]}; "
            "check that the layer index is valid."
        )

    maps = saliency_maps(capture, S, P, patch_start_idx, gw, gh)   # (S, gh, gw)
    heat = normalize_maps(maps, norm=args.norm, percentiles=tuple(args.percentile))

    if args.save_npz:
        np.savez(
            args.save_npz,
            maps=maps,
            heat=heat,
            layer=np.array(f"{layer_tag[0]}{layer_tag[1]}"),
            gw=gw, gh=gh, patch_size=patch_size,
        )
        print(f"Raw maps saved to {args.save_npz}")

    orig_rgbs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    if args.frame is not None:
        if not 0 <= args.frame < S:
            raise ValueError(f"--frame {args.frame} out of range [0, {S - 1}]")
        render_single(orig_rgbs[args.frame], heat[args.frame], transforms[args.frame], args.out,
                      cmap=args.cmap, alpha=args.alpha)
    else:
        render_grid(orig_rgbs, heat, transforms, args.out, cmap=args.cmap, alpha=args.alpha)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
