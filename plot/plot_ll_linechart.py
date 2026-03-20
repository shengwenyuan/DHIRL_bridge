"""
Compact line chart: Test LL vs (ns, na) combos, grouped by model type and hidden states.
Each model type gets a color family; hidden-state variants are lighter/darker shades.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Data: {(model_type, n_hidden, ns, na): [ll_values]} ──
# model_type decides color family, n_hidden decides shade within that family.
RAW_DATA = {
    # ("ModelA", n_hidden, ns, na): [ll1, ll2, ...]
    ("PGIQL",  32, 512, 32): [-2.1, -2.3, -2.0, -2.2, -2.15],
    ("PGIQL",  32, 1024, 32): [-1.8, -1.9, -1.85, -1.7, -1.82],
    ("PGIQL",  32, 2048, 32): [-1.5, -1.6, -1.55, -1.45, -1.52],
    ("PGIQL",  64, 512, 32): [-2.0, -2.1, -1.95, -2.05, -2.02],
    ("PGIQL",  64, 1024, 32): [-1.6, -1.7, -1.65, -1.55, -1.62],
    ("PGIQL",  64, 2048, 32): [-1.3, -1.4, -1.35, -1.25, -1.32],
    ("MaxEnt", 32, 512, 32): [-2.5, -2.6, -2.55, -2.45, -2.52],
    ("MaxEnt", 32, 1024, 32): [-2.2, -2.3, -2.25, -2.15, -2.22],
    ("MaxEnt", 32, 2048, 32): [-1.9, -2.0, -1.95, -1.85, -1.92],
    ("MaxEnt", 64, 512, 32): [-2.4, -2.5, -2.45, -2.35, -2.42],
    ("MaxEnt", 64, 1024, 32): [-2.0, -2.1, -2.05, -1.95, -2.02],
    ("MaxEnt", 64, 2048, 32): [-1.7, -1.8, -1.75, -1.65, -1.72],
    ("BCModel", 32, 512, 32): [-2.8, -2.9, -2.85, -2.75, -2.82],
    ("BCModel", 32, 1024, 32): [-2.5, -2.6, -2.55, -2.45, -2.52],
    ("BCModel", 32, 2048, 32): [-2.2, -2.3, -2.25, -2.15, -2.22],
    ("BCModel", 64, 512, 32): [-2.7, -2.8, -2.75, -2.65, -2.72],
    ("BCModel", 64, 1024, 32): [-2.3, -2.4, -2.35, -2.25, -2.32],
    ("BCModel", 64, 2048, 32): [-2.0, -2.1, -2.05, -1.95, -2.02],
}

# ── Color families: one hue per model type, shades for hidden states ──
# Base hues in HSL-like RGB anchors (R, G, B families)
COLOR_FAMILIES = {
    "PGIQL":   (0.15, 0.45, 0.85),   # blue family
    "MaxEnt":  (0.85, 0.25, 0.20),   # red family
    "BCModel": (0.20, 0.70, 0.30),   # green family
}


def shade_color(base_rgb, level, n_levels):
    """Shift brightness: level 0 = darkest, n_levels-1 = lightest."""
    t = 0.3 + 0.5 * (level / max(n_levels - 1, 1))
    return tuple(c * t + (1 - t) * 0.95 for c in base_rgb)


def build_series(raw_data):
    """Parse raw data into plottable series keyed by (model_type, n_hidden)."""
    series = defaultdict(dict)  # {(model, n_hidden): {x_label: [vals]}}
    for (model, n_hidden, ns, na), vals in raw_data.items():
        x_label = f"{ns}/{na}"
        series[(model, n_hidden)][x_label] = np.array(vals)
    return series


def main():
    series = build_series(RAW_DATA)

    # Sorted x-axis labels from all entries
    x_labels = sorted(
        {label for s in series.values() for label in s},
        key=lambda l: tuple(map(int, l.split('/')))
    )
    x_pos = np.arange(len(x_labels))

    # Discover model types and hidden-state counts
    model_types = sorted({k[0] for k in series})
    hidden_by_model = {m: sorted({k[1] for k in series if k[0] == m}) for m in model_types}

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in model_types:
        base_rgb = COLOR_FAMILIES.get(model, (0.5, 0.5, 0.5))
        hiddens = hidden_by_model[model]
        for lvl, nh in enumerate(hiddens):
            color = shade_color(base_rgb, lvl, len(hiddens))
            means, stds = [], []
            for label in x_labels:
                vals = series.get((model, nh), {}).get(label, np.array([np.nan]))
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            means, stds = np.array(means), np.array(stds)

            ax.plot(x_pos, means, marker='o', markersize=5, linewidth=2,
                    color=color, label=f"{model} (h={nh})")
            ax.fill_between(x_pos, means - stds, means + stds,
                            color=color, alpha=0.18)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel("num_states / num_actions", fontsize=13)
    ax.set_ylabel("Test Log-Likelihood", fontsize=13)
    ax.set_title("Test LL across (ns, na) by Model Type & Hidden States", fontsize=14)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot/ll_linechart.png", dpi=150)
    print("Saved plot/ll_linechart.png")
    plt.close()


if __name__ == '__main__':
    main()
