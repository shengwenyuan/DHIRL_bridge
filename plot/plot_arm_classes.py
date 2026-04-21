"""
Bridge K=4 latent-class filmstrip visualisation.

Each selected trajectory is rendered as a filmstrip of 48-px thumbnail frames.
Contiguous runs of the same class ("sub-sessions") are wrapped in a coloured
border box so the temporal segmentation is immediately visible.

Layout "horizontal" → one trajectory per row, frames go left→right.
Layout "vertical"   → one trajectory per column, frames go top→bottom.

Usage
-----
    python plot/plot_arm_classes.py
    # → produces plot/arm_classes.pdf
"""

import glob
import json
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ─── Configuration ──────────────────────────────────────────────────────────
FOLD_DIR      = "src_autotest/outputs_fresh/0411_K4_noreg/fold_0"
TRACEBACK_DIR = os.path.join(FOLD_DIR, "traceback_output")
K                  = 4
THUMB              = 94           # 21 × (94+6)/300 = 7.0" — exact A4 text width
MAX_FRAMES_PER_ROW = 21           # longest traj (51 frames) → 3 rows of ≤21
LAYOUT             = "horizontal" # "horizontal" | "vertical"
OUT                = "plot/arm_classes.pdf"
DPI                = 300          # print-quality; THUMB ≤ 256 avoids upsampling

# Ordered list of traj folder names to plot; None → all folders in TRACEBACK_DIR.
SELECTED = [
    # "traj_00000_idx7",        # Move the blue fork to the lower right burner
    # "traj_00006_idx33",       # (no language — scripted)
    "traj_00123_idx610",        # 1 place the pot lid next to the cloth
    "traj_00201_idx1013",       # 1 placed the green cube to the top of the yellow cube
    "traj_00225_idx1131",       # 1 put pan in sink
    # "traj_00400_idx1946",     # Move the green cloth to the back middle of the table
    "traj_00470_idx2265",       # 1 unfold the cloth from bottom right to top left
    "traj_00560_idx2747",       # 1 move the red pot to the left side of the sink
    # "traj_00600_idx2921",     # moved the banana to the upper middle of the table
    # "traj_00633_idx3044",     # fold the cloth from right to center
    # "traj_00666_idx3205",     # sweep into pile
    # "traj_00712_idx3408",     # move the blue cloth on the lower side of the left stove
    # "traj_00855_idx4070",     # move the yellow napkin to upper right of the table
    "traj_00913_idx4465",       # 1 Took and laundered in the washing machine
    "traj_00966_idx4680",       # 1 Move the pot from the front right to the back right corner.
]

CLASS_COLORS = [
    "#FF5722",  # vivid deep-orange — class 0
    "#2196F3",  # vivid blue        — class 1
    "#4CAF50",  # vivid green       — class 2
    "#E040FB",  # vivid magenta     — class 3
]
CLASS_NAMES = [          # display name for each class (used in legend + box labels)
    "CARRY",  # class 0
    "IDLE",
    "GRASP",
    "APPROACH/DEPART",
]
BOX_ALPHA = 0.95
BOX_LW    = 1.5
GAP       = 6    # px between thumbnails (at 300 DPI: 6/300" ≈ 0.5 mm)
# ────────────────────────────────────────────────────────────────────────────


def align_latents_bridge(fold_dir: str, K: int = 4) -> np.ndarray:
    """Identity permutation stub.

    Returns inv_perm where inv_perm[raw_z] = display_z.
    Single-fold outputs share the same model weights, so raw indices are
    already consistent across trajectories.  Swap this out when you need
    to compare across runs or different K settings.
    """
    return np.arange(K)


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_traj(traj_dir: str, inv_perm: np.ndarray, thumb: int) -> dict:
    """Load metadata and resized frames from a traceback trajectory directory."""
    summary      = _load_json(os.path.join(traj_dir, "summary.json"))
    raw_labels   = summary["class_labels"]
    language     = summary.get("language", "")

    display_labels = [int(inv_perm[k]) for k in raw_labels]

    frames = []
    for t, raw_k in enumerate(raw_labels):
        path = os.path.join(traj_dir, f"class_{raw_k}", f"frame_{t:03d}.png")
        img  = Image.open(path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        frames.append(np.array(img, dtype=np.uint8))

    return {
        "name":    os.path.basename(traj_dir),
        "language": language,
        "labels":  display_labels,
        "frames":  frames,
    }


def subsessions(labels: list) -> list:
    """Return [(start, end_inclusive, class_id), ...] for contiguous runs."""
    if not labels:
        return []
    result, start = [], 0
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            result.append((start, t - 1, labels[start]))
            start = t
    result.append((start, len(labels) - 1, labels[start]))
    return result


def make_filmstrip(frames: list, layout: str, thumb: int,
                   gap: int, max_per_line: int) -> np.ndarray:
    """Tile thumbnails into a wrapped grid with a gap between each frame.

    Canvas sized to fit exactly the real frames — no placeholder padding.
    Horizontal: lines stack top→bottom; Vertical: lines stack left→right.
    """
    T       = len(frames)
    slot    = thumb + gap
    n_lines = math.ceil(T / max_per_line)

    if layout == "horizontal":
        canvas = np.full((n_lines * slot, max_per_line * slot, 3), 255, dtype=np.uint8)
        for i, frame in enumerate(frames):
            r0 = (i // max_per_line) * slot
            c0 = (i %  max_per_line) * slot
            canvas[r0:r0 + thumb, c0:c0 + thumb] = frame
    else:
        canvas = np.full((max_per_line * slot, n_lines * slot, 3), 255, dtype=np.uint8)
        for i, frame in enumerate(frames):
            r0 = (i %  max_per_line) * slot
            c0 = (i // max_per_line) * slot
            canvas[r0:r0 + thumb, c0:c0 + thumb] = frame

    return canvas


def draw_boxes(ax, sessions: list, thumb: int, gap: int, layout: str,
               max_per_line: int):
    """Draw coloured border boxes tight around frame content, skipping the gap."""
    slot = thumb + gap
    for s, e, k in sessions:
        color = CLASS_COLORS[k % len(CLASS_COLORS)]
        t = s
        while t <= e:
            line    = t // max_per_line
            pos     = t % max_per_line
            seg_end = min(e, (line + 1) * max_per_line - 1)
            n       = seg_end - t + 1
            # width/height span exactly the frame pixels, gaps excluded
            box_w = (n - 1) * slot + thumb
            box_h = thumb
            if layout == "horizontal":
                rect = mpatches.Rectangle(
                    (pos * slot - 0.5, line * slot - 0.5), box_w, box_h,
                    linewidth=BOX_LW, edgecolor=color,
                    facecolor="none", alpha=BOX_ALPHA,
                )
            else:
                rect = mpatches.Rectangle(
                    (line * slot - 0.5, pos * slot - 0.5), box_h, box_w,
                    linewidth=BOX_LW, edgecolor=color,
                    facecolor="none", alpha=BOX_ALPHA,
                )
            ax.add_patch(rect)
            t = seg_end + 1


def main():
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    root          = os.path.normpath(os.path.join(script_dir, ".."))
    traceback_dir = os.path.join(root, TRACEBACK_DIR)
    fold_dir      = os.path.join(root, FOLD_DIR)
    out_path      = os.path.join(root, OUT)

    inv_perm = align_latents_bridge(fold_dir, K)

    if SELECTED is not None:
        traj_dirs = [os.path.join(traceback_dir, name) for name in SELECTED]
    else:
        traj_dirs = sorted(glob.glob(os.path.join(traceback_dir, "traj_*")))

    trajs        = [load_traj(d, inv_perm, THUMB) for d in traj_dirs]
    n            = len(trajs)
    n_lines_list = [math.ceil(len(t["frames"]) / MAX_FRAMES_PER_ROW) for t in trajs]

    slot = THUMB + GAP
    inch = 1.0 / DPI

    # Pixel-accurate layout: axes box = exactly image pixels → aspect="equal" never centers.
    # Key identity: axes_w_px = fig_w_px (full width), axes_h_px = nl*slot → 1 px = 1 data unit.
    TITLE_PX  = 50   # px above each subplot for title text  (≈ 8pt font + 3pt pad at DPI=300)
    GAP_PX    = 10   # px between consecutive subplot groups
    LEGEND_PX = 80   # px at figure bottom for legend
    TOP_PX    = 5    # px top margin

    if LAYOUT == "horizontal":
        fig_w_px    = MAX_FRAMES_PER_ROW * slot
        row_h_list  = [nl * slot for nl in n_lines_list]
        total_h_px  = (TOP_PX
                       + sum(TITLE_PX + rh for rh in row_h_list)
                       + (n - 1) * GAP_PX
                       + LEGEND_PX)
        fig = plt.figure(figsize=(fig_w_px * inch, total_h_px * inch))

        axes, y_px = [], TOP_PX
        for rh in row_h_list:
            y_px += TITLE_PX
            ax    = fig.add_axes([0.0,
                                  (total_h_px - y_px - rh) / total_h_px,
                                  1.0,
                                  rh / total_h_px])
            axes.append(ax)
            y_px += rh + GAP_PX

    else:
        fig_h_px   = MAX_FRAMES_PER_ROW * slot
        col_w_list = [nl * slot for nl in n_lines_list]
        total_w_px = (TOP_PX
                      + sum(TITLE_PX + cw for cw in col_w_list)
                      + (n - 1) * GAP_PX)
        fig = plt.figure(figsize=(total_w_px * inch,
                                  (fig_h_px + LEGEND_PX) * inch))
        legend_frac = LEGEND_PX / (fig_h_px + LEGEND_PX)
        axes, x_px = [], TOP_PX
        for cw in col_w_list:
            x_px += TITLE_PX
            ax    = fig.add_axes([x_px / total_w_px,
                                  legend_frac,
                                  cw / total_w_px,
                                  fig_h_px / (fig_h_px + LEGEND_PX)])
            axes.append(ax)
            x_px += cw + GAP_PX

    print(f"Figure: {fig.get_figwidth():.3f}\" × {fig.get_figheight():.3f}\"")
    for i, ax in enumerate(axes):
        p = ax.get_position()
        print(f"  ax[{i}] pos=({p.x0:.4f},{p.y0:.4f},{p.width:.4f},{p.height:.4f})"
              f" → display ({p.width*fig.get_figwidth()*DPI:.1f}"
              f" × {p.height*fig.get_figheight()*DPI:.1f}) px")

    for ax, traj, nl in zip(axes, trajs, n_lines_list):
        img      = make_filmstrip(traj["frames"], LAYOUT, THUMB, GAP, MAX_FRAMES_PER_ROW)
        sessions = subsessions(traj["labels"])
        img_h    = nl * slot
        img_w    = MAX_FRAMES_PER_ROW * slot

        ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.set_xlim(-0.5, img_w - 0.5)
        ax.set_ylim(img_h - 0.5, -0.5)
        draw_boxes(ax, sessions, THUMB, GAP, LAYOUT, MAX_FRAMES_PER_ROW)
        ax.set_axis_off()

        lang = (traj["language"] or "(no instruction)").strip()
        ax.set_title(lang, fontsize=8, loc="left", pad=3)

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
        for k in range(K)
    ]
    fig.legend(handles=legend_patches, ncol=K, loc="lower center",
               fontsize=7, framealpha=0.9)

    fig.savefig(out_path, dpi=DPI)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
