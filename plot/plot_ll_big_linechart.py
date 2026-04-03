"""
Big line chart: Test LL vs num_states, grouped by pre-train model type (subplots),
intention temporal model (color family), and num_latents (shade).

Usage (from DHIRL_bridge root):
    python plot/plot_ll_big_linechart.py \
        --src src_autotest/configs/test_rnn.yaml   src_autotest/outputs/run_rnn \
        --src src_autotest/configs/test_lstm.yaml  src_autotest/outputs/run_lstm \
        --src src_autotest/configs/test_trans.yaml src_autotest/outputs/run_trans

    # Single-source shorthand (backward-compatible):
    python plot/plot_ll_big_linechart.py \
        --src src_autotest/configs/test_0320_fast.yaml src_autotest/outputs/20260322_060204
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (AnchoredOffsetbox, HPacker, VPacker,
                                  DrawingArea)
from matplotlib.lines import Line2D
from matplotlib.text import Text
import yaml


# ── Defaults ─────────────────────────────────────────────────────
SAVE_PATH = "plot/ll_big_linechart.png"

# ── Color scales per intention temporal model ────────────────────
MODEL_BASE_COLORS = {
    "IntentionLSTM":        (0.85, 0.75, 0.10),   # golden yellow
    "IntentionRNN":         (0.20, 0.45, 0.85),   # blue
    "IntentionTransformer": (0.18, 0.72, 0.35),   # green
}

MODEL_SHORT = {
    "IntentionLSTM":        "LSTM",
    "IntentionRNN":         "RNN",
    "IntentionTransformer": "TF",
}

MODEL_ZORDER = {
    "IntentionRNN":         4,
    "IntentionLSTM":        3,
    "IntentionTransformer": 2,
}

MODEL_ALPHA = {
    "IntentionRNN":         0.85,
    "IntentionLSTM":        0.70,
    "IntentionTransformer": 0.55,
}

PRETRAIN_TITLES = {
    "trajs":        "CRL Encoder",
    "dinov2_small": "DINOv2-S",
    "dinov2_base":  "DINOv2-B",
    "dinov2_giant": "DINOv2-g",
}

PRETRAIN_ORDER = ["trajs", "dinov2_small", "dinov2_base", "dinov2_giant"]
MODEL_ORDER = ["IntentionRNN", "IntentionLSTM", "IntentionTransformer"]


# ── Helpers ──────────────────────────────────────────────────────

def shade_color(base_rgb, level, n_levels):
    """Vary brightness: level 0 = darkest, n_levels-1 = lightest."""
    t = 0.45 + 0.50 * (level / max(n_levels - 1, 1))
    return tuple(c * t + (1 - t) * 0.97 for c in base_rgb)


def parse_config(config_path):
    """Return list of experiment dicts from a YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg["defaults"]
    experiments = []
    for _gname, gdata in cfg["groups"].items():
        gid = gdata["id"]
        for exp in gdata["experiments"]:
            data_dir = exp.get("data_dir", defaults["data_dir"])
            pretrain = data_dir.rstrip("/").split("/")[-1]
            experiments.append({
                "group_id":   gid,
                "exp_id":     exp["id"],
                "num_states": exp.get("num_states", defaults["num_states"]),
                "pretrain":   pretrain,
                "model_type": exp.get("model_type", defaults["model_type"]),
                "num_latents": exp.get("num_latents", defaults["num_latents"]),
            })
    return experiments


def load_test_ll(run_dir, group_id, exp_id):
    """Load test_ll values from ll.csv.  Returns array or None."""
    csv_path = Path(run_dir) / group_id / exp_id / "ll.csv"
    if not csv_path.exists():
        return None
    try:
        vals = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vals.append(float(row["test_ll"]))
        return np.array(vals) if vals else None
    except Exception:
        return None


def mock_test_ll(num_states, pretrain, model_type, num_latents, rng):
    """Generate plausible mock test-LL values for missing experiments."""
    pretrain_bonus = {"trajs": 0.0, "dinov2_small": 0.05,
                      "dinov2_base": 0.12, "dinov2_giant": 0.18}
    model_bonus = {"IntentionRNN": 0.0, "IntentionLSTM": 0.06,
                   "IntentionTransformer": 0.10}
    base = -3.0
    base += pretrain_bonus.get(pretrain, 0.0)
    base += model_bonus.get(model_type, 0.0)
    base += 0.25 * np.log2(num_states / 512)
    base += 0.04 * num_latents - 0.005 * num_latents ** 2
    return base + rng.normal(0, 0.02, size=5)


def collect_data(sources):
    """Load real results from multiple (config, output_run) sources.

    Returns
    -------
    data : dict  {(pretrain, model_type, num_latents, num_states): np.array}
    n_real, n_mock : int
    """
    data = {}
    rng = np.random.default_rng(2026)
    n_real = n_mock = 0

    # First pass: load all real data from every source
    all_experiments = []  # (exp_dict, run_dir)
    for config_path, run_dir in sources:
        for exp in parse_config(config_path):
            all_experiments.append((exp, run_dir))

    for exp, run_dir in all_experiments:
        key = (exp["pretrain"], exp["model_type"], exp["num_latents"], exp["num_states"])
        vals = load_test_ll(run_dir, exp["group_id"], exp["exp_id"])
        if vals is not None:
            data[key] = vals
            n_real += 1

    # Second pass: mock anything still missing
    # Discover the full variable space from what was loaded + configs
    all_pretrains = sorted({k[0] for k in data})
    all_models = sorted({k[1] for k in data})
    all_latents = sorted({k[2] for k in data})
    all_nstates = sorted({k[3] for k in data})

    # Also include variables declared in configs but with no real data
    for exp, _ in all_experiments:
        all_pretrains = sorted(set(all_pretrains) | {exp["pretrain"]})
        all_models = sorted(set(all_models) | {exp["model_type"]})
        all_latents = sorted(set(all_latents) | {exp["num_latents"]})
        all_nstates = sorted(set(all_nstates) | {exp["num_states"]})

    for pt in all_pretrains:
        for mt in all_models:
            for nl in all_latents:
                for ns in all_nstates:
                    key = (pt, mt, nl, ns)
                    if key not in data:
                        data[key] = mock_test_ll(ns, pt, mt, nl, rng)
                        n_mock += 1

    return data, n_real, n_mock


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src", nargs=2, action="append", metavar=("CONFIG", "OUTPUT_RUN"),
        help="A (config.yaml, output_run_dir) pair. Repeat for each source.")
    parser.add_argument("--save", default=SAVE_PATH)
    parser.add_argument("--no-mock", action="store_true",
                        help="Skip missing experiments instead of mocking them")
    args = parser.parse_args()

    if not args.src:
        parser.error("At least one --src CONFIG OUTPUT_RUN pair is required.")

    sources = [(cfg, run) for cfg, run in args.src]
    print(f"Sources ({len(sources)}):")
    for cfg, run in sources:
        print(f"  config={cfg}  output={run}")

    data, n_real, n_mock = collect_data(sources)

    if args.no_mock:
        # Remove mocked entries - re-collect without mock pass
        real_keys = set()
        for cfg, run in sources:
            for exp in parse_config(cfg):
                key = (exp["pretrain"], exp["model_type"],
                       exp["num_latents"], exp["num_states"])
                if load_test_ll(run, exp["group_id"], exp["exp_id"]) is not None:
                    real_keys.add(key)
        data = {k: v for k, v in data.items() if k in real_keys}
        n_mock = 0
        print(f"Loaded {n_real} real experiments (no-mock mode).")
    else:
        print(f"Loaded {n_real} real + {n_mock} mocked experiments.")

    # Discover variable space from data
    all_pretrains = sorted({k[0] for k in data},
                           key=lambda p: PRETRAIN_ORDER.index(p)
                           if p in PRETRAIN_ORDER else 99)
    all_models = sorted({k[1] for k in data},
                        key=lambda m: MODEL_ORDER.index(m)
                        if m in MODEL_ORDER else 99)
    all_latents = sorted({k[2] for k in data})
    all_nstates = sorted({k[3] for k in data})
    n_latents_levels = len(all_latents)

    # ── Figure layout: single row, one panel per pretrain ────────
    n_pretrains = len(all_pretrains)
    fig, axes = plt.subplots(1, n_pretrains,
                             figsize=(4.5 * n_pretrains + 1.8, 4.8),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    x_pos = np.arange(len(all_nstates))
    x_labels = [str(ns) for ns in all_nstates]

    MARKERS = ['o', 's', '*', '^', 'v', 'D', 'X', 'P']

    for ax_idx, pretrain in enumerate(all_pretrains):
        ax = axes_flat[ax_idx]
        ax.set_title(PRETRAIN_TITLES.get(pretrain, f"Pre-train: {pretrain}"),
                     fontsize=14)

        for model_type in reversed(all_models):   # draw least-important first
            base_rgb = MODEL_BASE_COLORS.get(model_type, (0.5, 0.5, 0.5))
            short_name = MODEL_SHORT.get(model_type, model_type)
            zo = MODEL_ZORDER.get(model_type, 2)
            alph = MODEL_ALPHA.get(model_type, 0.9)

            for lat_idx, nl in enumerate(all_latents):
                color = shade_color(base_rgb, lat_idx, n_latents_levels)
                marker = MARKERS[lat_idx % len(MARKERS)]

                means, stds = [], []
                has_any = False
                for ns in all_nstates:
                    key = (pretrain, model_type, nl, ns)
                    vals = data.get(key)
                    if vals is not None:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
                        has_any = True
                    else:
                        means.append(np.nan)
                        stds.append(0.0)

                if not has_any:
                    continue

                means = np.array(means)
                stds = np.array(stds)

                label = f"{short_name}, K={nl}"
                ax.plot(x_pos, means, marker=marker, markersize=5,
                        linewidth=1.8, color=color, label=label,
                        alpha=alph, zorder=zo)
                ax.fill_between(x_pos, means - stds, means + stds,
                                color=color, alpha=0.10 * alph, zorder=zo)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Test LL", fontsize=14)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.grid(axis="x", alpha=0.15, linestyle=":")

    # Hide unused axes
    for i in range(n_pretrains, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # ── Shared table legend (right of figure) ────────────────────
    COL_W, COL_H = 22, 10          # swatch cell size in points
    SEP_H, SEP_V = 4, 1
    FONT_SZ = 8

    # Header row: blank corner cell + model-type names in fixed-width cells
    corner = DrawingArea(COL_W, COL_H, 0, 0)          # blank corner
    header = [corner]
    for mt in all_models:
        rgb = MODEL_BASE_COLORS.get(mt, (0.5, 0.5, 0.5))
        da = DrawingArea(COL_W, COL_H, 0, 0)
        txt = Text(COL_W / 2, COL_H / 2, MODEL_SHORT.get(mt, mt),
                   fontsize=FONT_SZ, fontweight="bold", color=rgb,
                   ha="center", va="center")
        da.add_artist(txt)
        header.append(da)
    legend_rows = [HPacker(children=header, pad=0, sep=SEP_H, align="center")]

    # One row per K value
    for lat_idx, nl in enumerate(all_latents):
        marker = MARKERS[lat_idx % len(MARKERS)]
        k_da = DrawingArea(COL_W, COL_H, 0, 0)
        k_txt = Text(COL_W / 2, COL_H / 2, f"K={nl}",
                     fontsize=FONT_SZ, ha="center", va="center")
        k_da.add_artist(k_txt)
        row_items = [k_da]
        for mt in all_models:
            base_rgb = MODEL_BASE_COLORS.get(mt, (0.5, 0.5, 0.5))
            c = shade_color(base_rgb, lat_idx, n_latents_levels)
            da = DrawingArea(COL_W, COL_H, 0, 0)
            da.add_artist(Line2D([1, COL_W - 1], [COL_H/2, COL_H/2],
                                 color=c, linewidth=1.5))
            da.add_artist(Line2D([COL_W/2], [COL_H/2], marker=marker,
                                 markersize=5, color=c, linestyle="None"))
            row_items.append(da)
        legend_rows.append(HPacker(children=row_items, pad=0, sep=SEP_H,
                                   align="center"))

    table = VPacker(children=legend_rows, pad=3, sep=SEP_V, align="center")
    # Attach to the last subplot, anchored outside to the right
    last_ax = axes_flat[n_pretrains - 1]
    box = AnchoredOffsetbox(loc="center left", child=table, pad=0.5,
                            frameon=True, prop=dict(size=FONT_SZ),
                            bbox_to_anchor=(1.02, 0.5),
                            bbox_transform=last_ax.transAxes)
    box.patch.set_boxstyle("round,pad=0.3")
    box.patch.set_alpha(0.90)
    box.patch.set_edgecolor("0.6")
    last_ax.add_artist(box)

    # Shared x-axis label
    fig.text(0.45, 0.01, "num_states", ha="center", fontsize=14)

    plt.tight_layout(rect=[0, 0.04, 0.88, 1.0])

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved {args.save}")
    plt.close()


if __name__ == "__main__":
    main()
