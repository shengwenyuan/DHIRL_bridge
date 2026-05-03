"""
Convergence loss curve for BridgeData V2.

Reads one or more E2.log files (one per random seed).  Each log contains
5 fold-runs.  Per seed the fold curves are averaged onto a shared iteration
grid; the grand mean ± std across all (seed × fold) runs is drawn as a bold
band.  When multiple seeds are given, each seed's per-fold mean is shown as a
thin semi-transparent line so run-to-run variability is visible.

Usage (from DHIRL_bridge root):
    # single seed
    python plot/plot_convergence_bridge.py \
        --log src_autotest/logs/20260410_174341/G01/E2.log \
        --out plot/convergence_bridge.pdf

    # multiple seeds
    python plot/plot_convergence_bridge.py \
        --log src_autotest/logs/seed42/G01/E2.log \
        --log src_autotest/logs/seed0/G01/E2.log  \
        --log src_autotest/logs/seed7/G01/E2.log  \
        --out plot/convergence_bridge.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
matplotlib.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size':        9,
    'axes.labelsize':  10,
    'axes.titlesize':   9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize':  9,
})


# ── Style ────────────────────────────────────────────────────────────────
BLUE         = (0.20, 0.45, 0.85)
GRAY_THRESH  = (0.50, 0.50, 0.50)

# One accent color per seed (cycles if more seeds than colors)
SEED_COLORS = [
    (0.20, 0.45, 0.85),   # seed 0 – blue  (matches IntentionRNN style)
    (0.85, 0.35, 0.10),   # seed 1 – orange
    (0.15, 0.62, 0.35),   # seed 2 – green
    (0.65, 0.18, 0.58),   # seed 3 – purple
    (0.60, 0.52, 0.08),   # seed 4 – olive
]

SEED_LINE_ALPHA  = 0.45   # per-seed mean curve
BAND_ALPHA       = 0.15   # grand std band fill
GRAND_MEAN_ALPHA = 0.95


# ── Log parser ───────────────────────────────────────────────────────────

def parse_log(log_path):
    """Return list of dicts (one per fold-run) with keys:
       iters (ndarray), losses (ndarray), conv_iter, conv_loss, total_time.
    """
    runs = []
    cur_iters, cur_losses = [], []

    iter_re = re.compile(r"^Iteration (\d+), Loss: ([0-9.]+),")
    conv_re = re.compile(
        r"^Iteration (\d+), Converged with Loss: ([0-9.]+), Total time: ([0-9.]+)s")

    with open(log_path) as f:
        for line in f:
            m = conv_re.match(line)
            if m:
                conv_iter  = int(m.group(1))
                conv_loss  = float(m.group(2))
                total_time = float(m.group(3))
                if not cur_iters or cur_iters[-1] != conv_iter:
                    cur_iters.append(conv_iter)
                    cur_losses.append(conv_loss)
                runs.append({
                    "iters":      np.array(cur_iters),
                    "losses":     np.array(cur_losses),
                    "conv_iter":  conv_iter,
                    "conv_loss":  conv_loss,
                    "total_time": total_time,
                })
                cur_iters, cur_losses = [], []
                continue
            m = iter_re.match(line)
            if m:
                cur_iters.append(int(m.group(1)))
                cur_losses.append(float(m.group(2)))

    return runs


def runs_to_logmat(runs, grid):
    """Interpolate each run's loss curve (log-space) onto `grid`."""
    mat = []
    for r in runs:
        log_l = np.log(r["losses"])
        interp = np.interp(grid, r["iters"], log_l,
                           left=log_l[0], right=log_l[-1])
        mat.append(interp)
    return np.array(mat)          # (n_runs, len(grid))


# ── Plot ─────────────────────────────────────────────────────────────────

def plot(all_runs_per_seed, out_path, loss_threshold=1e-3):
    """
    all_runs_per_seed : list of lists  [ [run, run, ...], [run, run, ...], ... ]
                        outer index = seed, inner index = fold
    """
    n_seeds   = len(all_runs_per_seed)
    max_iter  = max(r["conv_iter"]
                    for seed_runs in all_runs_per_seed
                    for r in seed_runs)
    grid      = np.arange(4, max_iter + 1, 4)

    # ---------- compute per-seed mean curves and grand statistics ----------
    seed_means = []          # (n_seeds, len(grid)) in log space
    all_log_mats = []        # flattened across seeds

    for seed_runs in all_runs_per_seed:
        lmat = runs_to_logmat(seed_runs, grid)   # (n_folds, len(grid))
        seed_means.append(lmat.mean(axis=0))
        all_log_mats.append(lmat)

    seed_means   = np.array(seed_means)                   # (n_seeds, len(grid))
    all_log_flat = np.vstack(all_log_mats)                # (n_seeds*n_folds, len(grid))

    grand_mean = all_log_flat.mean(axis=0)
    grand_std  = all_log_flat.std(axis=0)

    # ---------- figure --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # Per-seed mean curves (only drawn when there are multiple seeds)
    if n_seeds > 1:
        for s_idx, s_mean_log in enumerate(seed_means):
            c = SEED_COLORS[s_idx % len(SEED_COLORS)]
            ax.semilogy(grid, np.exp(s_mean_log),
                        color=c, linewidth=1.2, alpha=SEED_LINE_ALPHA,
                        zorder=3)

    # Grand mean ± std band
    grand_lin = np.exp(grand_mean)
    lo_lin    = np.exp(grand_mean - grand_std)
    hi_lin    = np.exp(grand_mean + grand_std)
    ax.fill_between(grid, lo_lin, hi_lin,
                    color=BLUE, alpha=BAND_ALPHA, zorder=4)
    ax.semilogy(grid, grand_lin,
                color=BLUE, linewidth=2.2, alpha=GRAND_MEAN_ALPHA,
                zorder=5)

    # Convergence threshold
    ax.axhline(loss_threshold,
               color=GRAY_THRESH, linewidth=0.9, linestyle="--", zorder=2)

    # Axes
    ax.set_xlim(left=0, right=max_iter + 4)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("EM objective (loss)")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.grid(axis="y", which="both", alpha=0.20, linestyle="--")
    ax.grid(axis="x", alpha=0.12, linestyle=":")

    # Legend
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2.2,
               alpha=GRAND_MEAN_ALPHA,
               label="mean ± std" + (f" ({n_seeds} seeds × {len(all_runs_per_seed[0])} folds)"
                                     if n_seeds > 1 else
                                     f" ({len(all_runs_per_seed[0])} folds)")),
    ]
    if n_seeds > 1:
        handles.append(
            Line2D([0], [0], color=GRAY_THRESH, linewidth=1.2,
                   alpha=SEED_LINE_ALPHA, label="per-seed mean"))
    handles.append(
        Line2D([0], [0], color=GRAY_THRESH, linewidth=0.9, linestyle="--",
               label=f"conv. threshold (1e{int(np.log10(loss_threshold))})"))
    ax.legend(handles=handles, loc="upper right",
              framealpha=0.85, edgecolor="0.7")

    plt.tight_layout(pad=1.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", required=True, metavar="LOG",
                        help="Path to E2.log.  Repeat for each random seed.")
    parser.add_argument("--out", default="plot/convergence_bridge.pdf")
    parser.add_argument("--loss-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    all_runs_per_seed = []
    for log_path in args.log:
        runs = parse_log(log_path)
        print(f"{log_path}: {len(runs)} fold-runs, "
              f"conv_iters={[r['conv_iter'] for r in runs]}")
        all_runs_per_seed.append(runs)

    plot(all_runs_per_seed, args.out, args.loss_threshold)


if __name__ == "__main__":
    main()
