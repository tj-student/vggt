#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render VGGT-Det saliency maps for all f0..f23 or all g0..g23 blocks.

Examples
--------
python visualize_attention_vggtdet_all_layers.py --images examples/room/images --block f
python visualize_attention_vggtdet_all_layers.py --images examples/room/images --block g
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the single-layer script's preprocessing, model loading, and coordinate
# mapping so that a panel here has exactly the same meaning as its output.
import visualize_attention_vggtdet as base


class AllLayersAttentionCapture:
    """Capture VGGT-Det column-mean saliency for every block of one type."""

    ROW_CHUNK = 256

    def __init__(self, aggregator, kind, num_frames, tokens_per_frame):
        self.aggregator = aggregator
        self.kind = kind
        self.num_frames = int(num_frames)
        self.P = int(tokens_per_frame)
        self._current = None
        self._installed = []
        self._orig_sdpa = None
        self.accumulators = {}  # layer index -> (B, N) CPU float32 column sums
        self.head_counts = {}
        self.sequence_lengths = {}

    def __enter__(self):
        blocks = self.aggregator.frame_blocks if self.kind == "frame" else self.aggregator.global_blocks
        for layer_idx, block in enumerate(blocks):
            attn = block.attn
            original_forward = attn.forward

            def tagged_forward(x, pos=None, _original=original_forward, _index=layer_idx):
                previous, self._current = self._current, _index
                try:
                    return _original(x, pos=pos)
                finally:
                    self._current = previous

            attn.forward = tagged_forward
            self._installed.append(attn)

        self._orig_sdpa = F.scaled_dot_product_attention

        def sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if self._current is not None and query.dim() == 4:
                self._capture(self._current, query, key)
            return self._orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, **kwargs
            )

        F.scaled_dot_product_attention = sdpa_wrapper
        return self

    def __exit__(self, *exc):
        for attn in self._installed:
            attn.__dict__.pop("forward", None)
        self._installed = []
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None
        self._current = None

    @torch.no_grad()
    def _capture(self, layer_idx, q, k):
        batch, heads, sequence, dim = q.shape
        expected_batch = self.num_frames if self.kind == "frame" else 1
        expected_sequence = self.P if self.kind == "frame" else self.num_frames * self.P
        if batch != expected_batch or sequence != expected_sequence:
            raise RuntimeError(
                f"Unexpected {self.kind} attention shape {(batch, sequence)}; "
                f"expected {(expected_batch, expected_sequence)}"
            )

        key32 = k.float()
        accumulator = torch.zeros(batch, sequence, device=q.device, dtype=torch.float32)
        for start in range(0, sequence, self.ROW_CHUNK):
            rows = slice(start, min(start + self.ROW_CHUNK, sequence))
            logits = torch.matmul(q[:, :, rows, :].float(), key32.transpose(-2, -1)) * (dim ** -0.5)
            # Sum, rather than retain, query rows and heads: this is the exact
            # numerator of attention.mean(heads).mean(query_rows).
            accumulator += logits.softmax(dim=-1).sum(dim=1).sum(dim=1)
        self.accumulators[layer_idx] = accumulator.cpu()
        self.head_counts[layer_idx] = heads
        self.sequence_lengths[layer_idx] = sequence

    def maps(self, patch_start_idx, grid_width, grid_height):
        """Return layer index -> (S, gh, gw) raw per-frame saliency maps."""
        expected_patches = grid_width * grid_height
        output = {}
        for layer_idx, accumulator in self.accumulators.items():
            mean = accumulator / (self.head_counts[layer_idx] * self.sequence_lengths[layer_idx])
            if self.kind == "frame":
                patches = mean[:, patch_start_idx:]
            else:
                patches = mean.view(self.num_frames, self.P)[:, patch_start_idx:]
            if patches.shape[1] != expected_patches:
                raise RuntimeError(f"Layer {layer_idx}: expected {expected_patches} patch tokens, got {patches.shape[1]}")
            output[layer_idx] = patches.reshape(self.num_frames, grid_height, grid_width).numpy()
        return output


def render_overview(originals, transforms, maps, out_path, block, cmap, alpha):
    """Render one row per layer and one column per input image."""
    layer_count = len(maps)
    frame_count = len(originals)
    fig, axes = base.plt.subplots(
        layer_count, frame_count,
        figsize=(3.2 * frame_count, 2.6 * layer_count),
        squeeze=False,
        constrained_layout=True,
    )
    for layer_idx in range(layer_count):
        # Normalization remains independent for each image/layer, matching the
        # original single-layer visualizer's per-frame normalization behavior.
        heatmaps = base.normalize_maps(maps[layer_idx])
        for frame_idx in range(frame_count):
            ax = axes[layer_idx, frame_idx]
            x0, x1, y0, y1 = base._overlay_extent(transforms[frame_idx])
            ax.imshow(originals[frame_idx])
            ax.imshow(
                heatmaps[frame_idx], cmap=cmap, alpha=alpha,
                interpolation="bilinear", origin="upper", extent=(x0, x1, y1, y0),
            )
            if layer_idx == 0:
                ax.set_title(f"Input {frame_idx}", fontsize=12)
            if frame_idx == 0:
                ax.set_ylabel(f"{block}{layer_idx}", fontsize=12, rotation=0, labelpad=24, va="center")
            ax.axis("off")
    fig.suptitle(f"VGGT-Det attention saliency — {block}0 to {block}{layer_count - 1}", fontsize=16)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    base.plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Render all 24 VGGT-Det frame-wise or global attention layers.")
    parser.add_argument("--images", nargs="+", required=True, help="Image files and/or frame folders")
    parser.add_argument("--block", choices=("f", "g"), default="f", help="f: f0..f23 frame-wise; g: g0..g23 global")
    parser.add_argument("--mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--weights", default=None, help="Local VGGT checkpoint; otherwise facebook/VGGT-1B")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--out", default=None, help="Output PNG (default: attention_all_f.png or attention_all_g.png)")
    parser.add_argument("--save-npz", default=None, help="Optionally save all raw maps")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = base.collect_image_paths(args.images)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"auto": None, "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    images_np, transforms = base.preprocess_images(paths, mode=args.mode)
    frames, height, width, _ = images_np.shape
    model = base.load_model(args.weights, device, dtype)
    aggregator = model.aggregator
    kind = "frame" if args.block == "f" else "global"
    patch_start_idx = int(aggregator.patch_start_idx)
    grid_width, grid_height = width // 14, height // 14
    tokens_per_frame = patch_start_idx + grid_width * grid_height
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype)
    print(f"Capturing {args.block}0..{args.block}{len(aggregator.frame_blocks) - 1} for {frames} frame(s) ...")
    with torch.no_grad(), AllLayersAttentionCapture(aggregator, kind, frames, tokens_per_frame) as capture:
        aggregator(images)
    maps = capture.maps(patch_start_idx, grid_width, grid_height)
    expected = len(aggregator.frame_blocks)
    if len(maps) != expected:
        raise RuntimeError(f"Captured {len(maps)} layers, expected {expected}")
    out_path = args.out or f"attention_all_{args.block}.png"
    originals = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    render_overview(originals, transforms, maps, out_path, args.block, args.cmap, args.alpha)
    if args.save_npz:
        np.savez(args.save_npz, **{f"{args.block}{index}": value for index, value in maps.items()},
                 gw=grid_width, gh=grid_height, patch_size=14)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
