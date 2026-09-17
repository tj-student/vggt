#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention-map visualization for VGGT (VGGT-Det paper style).

For a chosen frame and one or more query boxes (in original image pixel
coordinates), it produces a two-panel figure:

    left  = input image with the green query box(es)
    right = attention heatmap from the query-box patch tokens, overlaid on the
            image with a jet-style colormap (blue = low, red = high)

How it works
------------
VGGT's attention runs through fused kernels (F.scaled_dot_product_attention)
that never materialize the attention matrix, so the weights cannot be grabbed
with a plain forward hook. Instead this script:

  1. wraps the forward of every selected aggregator attention module (frame and
     global blocks) to tag "which layer is running",
  2. monkey-patches torch.nn.functional.scaled_dot_product_attention during the
     forward pass: when a tagged layer calls it, the wrapper recomputes the
     attention matrix only for the query rows that correspond to the query-box
     patch tokens (in float32), applies softmax, and averages over heads,
  3. slices the resulting attention distribution back to the patch tokens of
     the query frame, reshapes it to the patch grid, averages over the selected
     layers, and overlays it on the image.

Token layout per frame (see vggt/models/aggregator.py):
    [camera (1), register (num_register_tokens), patches (gh*gw, row-major)]
so patch_start_idx = 1 + num_register_tokens = 5 for VGGT-1B.

Examples
--------
    python visualize_attention.py --images examples/room/images --query-frame 0 \
        --box 230 300 420 480 --out room_attn.png

    # average the last four global layers, two query boxes, save raw maps:
    python visualize_attention.py --images examples/kitchen/images --layers g20,g21,g22,g23 \
        --box 200 380 340 480 --box 60 120 200 300 --save-npz kitchen_attn.npz

    # attention averaged over ALL global layers:
    python visualize_attention.py --images examples/room/images --layers all-global \
        --box 230 300 420 480
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
from matplotlib.patches import Rectangle
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
    transform needed to map boxes and heatmaps between the two coordinate systems.

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
            # width -> target_size, height scaled (multiple of patch_size), center-crop if needed
            new_w = target_size
            new_h = round(h * (new_w / w) / patch_size) * patch_size
            im2 = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            top = (new_h - target_size) // 2 if new_h > target_size else 0
            if new_h > target_size:
                im2 = im2.crop((0, top, new_w, top + target_size))
            t = dict(scale_x=target_size / w, scale_y=new_h / h, off_x=0.0, off_y=-float(top))
        else:  # pad
            # largest dimension -> target_size, smaller scaled (multiple of patch_size), pad to square
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


def box_to_patch_range(box, t, patch_size=14):
    """Map a box in original-image pixels to flat patch indices on the processed grid."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    gw = t["proc_w"] // patch_size
    gh = t["proc_h"] // patch_size
    if gw * gh <= 0:
        raise ValueError(f"Processed size {t['proc_w']}x{t['proc_h']} is too small for patch size {patch_size}")

    px1 = x1 * t["scale_x"] + t["off_x"]
    py1 = y1 * t["scale_y"] + t["off_y"]
    px2 = x2 * t["scale_x"] + t["off_x"]
    py2 = y2 * t["scale_y"] + t["off_y"]

    ix1 = max(int(np.floor(px1 / patch_size)), 0)
    iy1 = max(int(np.floor(py1 / patch_size)), 0)
    ix2 = min(int(np.ceil(px2 / patch_size)) - 1, gw - 1)
    iy2 = min(int(np.ceil(py2 / patch_size)) - 1, gh - 1)
    if ix2 < ix1 or iy2 < iy1:
        raise ValueError(
            f"Query box {list(box)} does not cover any patch on the {gw}x{gh} grid; "
            "check the coordinates (they must be in the ORIGINAL image pixel coordinates)."
        )

    ys, xs = np.meshgrid(np.arange(iy1, iy2 + 1), np.arange(ix1, ix2 + 1), indexing="ij")
    return (ys * gw + xs).reshape(-1), (gw, gh)


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Instruments an Aggregator so that, during a forward pass, the attention
    distribution from selected query rows (the query-box patch tokens) is
    captured for the requested layers, averaged over heads: tag -> (R, N) fp32.

    Only rows of the attention matrix are ever materialized (in chunks), so the
    memory overhead is independent of using the full matrix.
    """

    ROW_CHUNK = 128  # query rows per fp32 matmul chunk

    def __init__(self, aggregator, wanted_layers, query_frame, num_frames, P,
                 patch_start_idx, patch_flat_idx):
        self.aggregator = aggregator
        self.wanted = set(map(tuple, wanted_layers))
        self.query_frame = int(query_frame)
        self.num_frames = int(num_frames)
        self.P = int(P)
        self.patch_start_idx = int(patch_start_idx)
        self.patch_flat_idx = np.asarray(patch_flat_idx, dtype=np.int64)

        self._current = None
        self._installed = []      # (attn_module, had_instance_forward)
        self._orig_sdpa = None
        self._warned = set()
        self.maps = {}            # (kind, layer_idx) -> torch.Tensor (R, N) on CPU

    def __enter__(self):
        self._install_block_hooks()
        self._patch_sdpa()
        return self

    def __exit__(self, *exc):
        self._restore()

    # -- installation -------------------------------------------------------

    def _install_block_hooks(self):
        for group, kind in (("frame_blocks", "frame"), ("global_blocks", "global")):
            blocks = getattr(self.aggregator, group)
            for i, block in enumerate(blocks):
                tag = (kind, i)
                if tag not in self.wanted:
                    continue
                attn = block.attn
                orig_forward = attn.forward
                capture = self

                def tagged_forward(x, pos=None, _orig=orig_forward, _tag=tag):
                    prev, capture._current = capture._current, _tag
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
                capture._capture(capture._current, query, key)
            return capture._orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
            )

        F.scaled_dot_product_attention = sdpa_wrapper

    def _restore(self):
        # drop the instance-level forward so the class method takes over again
        for attn, _orig in self._installed:
            attn.__dict__.pop("forward", None)
        self._installed = []
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None
        self._current = None

    # -- capture ------------------------------------------------------------

    @torch.no_grad()
    def _capture(self, tag, q, k):
        B, H, N, D = q.shape
        kind, _ = tag
        # frame-wise blocks run on (B*S, P, C): the query frame is one batch element;
        # global blocks run on (B, S*P, C) with scene batch size 1.
        if kind == "frame":
            expected_batch, batch_idx, expected_N = self.num_frames, self.query_frame, self.P
        else:
            expected_batch, batch_idx, expected_N = 1, 0, self.num_frames * self.P
        if B != expected_batch:
            raise RuntimeError(
                f"Attention visualization requires scene batch size 1 (got batch {B}, expected {expected_batch})"
            )

        if N != expected_N:
            if tag not in self._warned:
                print(f"[attention-vis] skipping {tag}: sequence length {N} != expected {expected_N}")
                self._warned.add(tag)
            return

        base = self.patch_start_idx if kind == "frame" else self.query_frame * self.P + self.patch_start_idx
        rows = torch.as_tensor(self.patch_flat_idx, device=q.device) + base

        k32 = k[batch_idx : batch_idx + 1].float()
        chunks = []
        for s in range(0, rows.numel(), self.ROW_CHUNK):
            r = rows[s : s + self.ROW_CHUNK]
            q_sel = q[batch_idx : batch_idx + 1, :, r, :].float()
            logits = torch.matmul(q_sel, k32.transpose(-2, -1)) * (D ** -0.5)
            attn = logits.softmax(dim=-1).mean(dim=1)[0]  # (r, N), heads averaged
            chunks.append(attn.cpu())
        self.maps.setdefault(tag, []).append(torch.cat(chunks, dim=0))


# ---------------------------------------------------------------------------
# Map post-processing / rendering
# ---------------------------------------------------------------------------

def parse_layers(spec, depth):
    """Parse a layer spec like 'g23', 'g20,g21,g22,g23', 'all-global', 'all'."""
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return [("frame", i) for i in range(depth)] + [("global", i) for i in range(depth)]
    tags = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part == "all-global":
            tags += [("global", i) for i in range(depth)]
        elif part == "all-frame":
            tags += [("frame", i) for i in range(depth)]
        elif len(part) >= 2 and part[0] in ("f", "g") and part[1:].isdigit():
            idx = int(part[1:])
            if not 0 <= idx < depth:
                raise ValueError(f"Layer index {idx} out of range [0, {depth - 1}]")
            tags.append(("frame" if part[0] == "f" else "global", idx))
        else:
            raise ValueError(f"Cannot parse layer '{part}' (expected e.g. g23, f12, all-global, all)")
    if not tags:
        raise ValueError("Empty layer spec")
    return tags


def layer_patch_maps(capture, query_frame, P, patch_start_idx, gw, gh):
    """tag -> (gh, gw) numpy attention map over the query frame's patches."""
    npatch = gw * gh
    out = {}
    for tag, mats in capture.maps.items():
        vec = torch.cat(mats, dim=0).mean(dim=0).numpy()  # (N,)
        kind, _ = tag
        if kind == "frame":
            keys = vec[patch_start_idx : patch_start_idx + npatch]
        else:
            s = query_frame * P + patch_start_idx
            keys = vec[s : s + npatch]
        out[tag] = keys.reshape(gh, gw)
    return out


def normalize_map(m, percentiles=(2.0, 98.0)):
    lo, hi = np.percentile(m, percentiles)
    if hi - lo < 1e-12:
        return np.zeros_like(m)
    return np.clip((m - lo) / (hi - lo), 0.0, 1.0)


def render_figure(orig_rgb, boxes, heatmap, t, out_path, cmap="jet", alpha=0.55,
                  title_left="Input Images", title_right="Attention Visualization"):
    """Two-panel figure: original image + green boxes | original image + heatmap overlay."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    ax_l.imshow(orig_rgb)
    ax_l.set_title(title_left, fontsize=15)
    ax_l.axis("off")
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        ax_l.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2.5)
        )

    ax_r.imshow(orig_rgb)
    ax_r.set_title(title_right, fontsize=15)
    ax_r.axis("off")
    # processed-image rectangle mapped back to original pixel coordinates
    x0 = (0.0 - t["off_x"]) / t["scale_x"]
    x1 = (t["proc_w"] - t["off_x"]) / t["scale_x"]
    y0 = (0.0 - t["off_y"]) / t["scale_y"]
    y1 = (t["proc_h"] - t["off_y"]) / t["scale_y"]
    ax_r.imshow(
        heatmap, cmap=cmap, alpha=alpha, interpolation="bilinear",
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

def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT attention-map visualization (VGGT-Det paper style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", nargs="+", required=True,
                   help="Image files and/or folders (frames of one scene, sorted by name)")
    p.add_argument("--query-frame", type=int, default=0,
                   help="Index of the frame the query box refers to")
    p.add_argument("--box", nargs=4, type=float, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                   help="Query box in ORIGINAL image pixel coordinates (repeatable; "
                        "attention is averaged over the union of all boxes)")
    p.add_argument("--layers", default="g23",
                   help="Attention layers to average, e.g. 'g23', 'g20,g21,g22,g23', "
                        "'f12', 'all-global', 'all' (f=frame-wise, g=global block, 0..23)")
    p.add_argument("--mode", choices=("crop", "pad"), default="crop",
                   help="Preprocessing mode, same as load_and_preprocess_images")
    p.add_argument("--weights", default=None,
                   help="Path to a local VGGT checkpoint (state dict). Default: download facebook/VGGT-1B")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    p.add_argument("--cmap", default="jet", help="Matplotlib colormap for the heatmap")
    p.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity")
    p.add_argument("--percentile", nargs=2, type=float, default=(2.0, 98.0), metavar=("LO", "HI"),
                   help="Percentile clip for heatmap normalization")
    p.add_argument("--out", default="attention_vis.png", help="Output figure path")
    p.add_argument("--save-npz", default=None, help="Optionally save the raw per-layer patch maps (.npz)")
    p.add_argument("--list-layers", action="store_true", help="Print available layer tags and exit")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"auto": None, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    paths = collect_image_paths(args.images)
    print(f"Using {len(paths)} frames, query frame {args.query_frame}")
    if not 0 <= args.query_frame < len(paths):
        raise ValueError(f"--query-frame {args.query_frame} out of range [0, {len(paths) - 1}]")
    if not args.box:
        raise ValueError("Please provide at least one query box via --box X1 Y1 X2 Y2")

    images_np, transforms = preprocess_images(paths, mode=args.mode)
    S, H, W, _ = images_np.shape
    patch_size = 14

    if args.list_layers:
        print("VGGT-1B default: depth=24 -> f0..f23 (frame-wise blocks), g0..g23 (global blocks)")
        print("e.g. --layers g23, --layers g20,g21,g22,g23, --layers all-global, --layers all")
        return

    model = load_model(args.weights, device, dtype)
    agg = model.aggregator

    depth = len(agg.frame_blocks)
    wanted = parse_layers(args.layers, depth)
    print(f"Layers to average: {[f'{k[0]}{i}' for k, i in wanted]}")

    patch_start_idx = int(agg.patch_start_idx)          # 1 + num_register_tokens (=5 for VGGT-1B)
    gw, gh = W // patch_size, H // patch_size
    P = patch_start_idx + gw * gh
    print(f"Grid {gw}x{gh} patches, P={P} tokens/frame, S={S} frames")

    patch_flat, _ = box_to_patch_range(args.box[0], transforms[args.query_frame], patch_size)
    for extra in args.box[1:]:
        more, _ = box_to_patch_range(extra, transforms[args.query_frame], patch_size)
        patch_flat = np.concatenate([patch_flat, more])
    print(f"Query box(es) cover {patch_flat.size} patch tokens")

    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)  # (1, S, 3, H, W)

    capture = AttentionCapture(
        agg, wanted, query_frame=args.query_frame, num_frames=S, P=P,
        patch_start_idx=patch_start_idx, patch_flat_idx=patch_flat,
    )
    print("Running aggregator forward with attention capture ...")
    with torch.no_grad(), capture:
        agg(images)

    missing = [f"{k}{i}" for k, i in wanted if (k, i) not in capture.maps]
    if missing:
        print(f"Warning: no attention captured for layers {missing}")
    maps = layer_patch_maps(capture, args.query_frame, P, patch_start_idx, gw, gh)
    combined = np.mean([maps[tag] for tag in maps], axis=0)
    heat = normalize_map(combined, percentiles=tuple(args.percentile))

    if args.save_npz:
        np.savez(
            args.save_npz,
            combined=combined,
            heat=heat,
            gw=gw, gh=gh, patch_size=patch_size,
            query_frame=args.query_frame,
            boxes=np.array(args.box, dtype=np.float32),
            **{f"{k}{i}": m for (k, i), m in maps.items()},
        )
        print(f"Raw maps saved to {args.save_npz}")

    orig = Image.open(paths[args.query_frame]).convert("RGB")
    orig_rgb = np.asarray(orig)
    render_figure(orig_rgb, args.box, heat, transforms[args.query_frame], args.out,
                  cmap=args.cmap, alpha=args.alpha)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
