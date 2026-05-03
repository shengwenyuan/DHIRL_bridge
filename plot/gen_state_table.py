"""
Generate a LaTeX table: discretization statistics vs num_states,
grouped by visual encoding model (rows).

Scans data_autotest/<subdir>/trajs_<NS>_<NA>.json for available data.
Blank cells where data is missing.

Usage (from DHIRL_bridge root):
    python plot/gen_state_table.py
    python plot/gen_state_table.py --num_states 512 1024 1536 2048
    python plot/gen_state_table.py --save plot/state_table.tex
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

# ── Visual encoder directories and display names ────────────────
ENCODERS = [
    ("complete",     "CRL Encoder"),
    ("dinov2_small", "DINOv2-S"),
    ("dinov2_base",  "DINOv2-B"),
    ("dinov2_giant", "DINOv2-g"),
]

DATA_ROOT = Path("data_autotest")
NUM_ACTIONS = 32  # fixed across experiments

METRICS = [
    ("coverage",        r"Coverage (\%)"),
    ("avg_occ",         r"Avg.\ revisits"),
    ("singleton_ratio", r"Singleton (\%)"),
]


# ── Stats (same logic as count_times.py) ────────────────────────
def compute_stats(trajs, num_states):
    records = {'coverage': [], 'avg_occ': [], 'singleton_ratio': []}
    for traj in trajs:
        arr = np.array(traj)
        states = arr[:, 0].astype(int)
        counter = Counter(states.tolist())
        counts = np.array(list(counter.values()))
        n_unique = len(counter)
        records['coverage'].append(n_unique / num_states)
        records['avg_occ'].append(np.mean(counts))
        records['singleton_ratio'].append(np.sum(counts == 1) / n_unique)
    return {k: (np.mean(v), np.std(v)) for k, v in records.items()}


def load_trajs(subdir, ns):
    path = DATA_ROOT / subdir / f"trajs_{ns}_{NUM_ACTIONS}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_val(metric, mean, std):
    if metric == "coverage":
        return f"{mean*100:.2f}{{\\scriptstyle\\pm{std*100:.2f}}}"
    elif metric == "singleton_ratio":
        return f"{mean*100:.1f}{{\\scriptstyle\\pm{std*100:.1f}}}"
    else:  # avg_occ
        return f"{mean:.1f}{{\\scriptstyle\\pm{std:.1f}}}"


def generate_latex(num_states_list):
    n_ns = len(num_states_list)
    n_metrics = len(METRICS)

    col_spec = "ll" + "c" * n_ns
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Effect of $k$-means state granularity on trajectory discretization statistics (mean$\pm$std across trajectories). CRL statistics are pending.}")
    lines.append(r"\label{tab:state_granularity}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Two-row header
    lines.append(r"\multirow{2}{*}{\textbf{Encoder}} & \multirow{2}{*}{\textbf{Metric}}")
    lines.append(r"    & \multicolumn{" + str(n_ns) + r"}{c}{\textbf{Number of states} $|\mathcal{S}|$} \\")
    lines.append(r"\cmidrule(lr){3-" + str(2 + n_ns) + "}")
    header2 = "    & "
    for ns in num_states_list:
        header2 += f" & {ns}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\hline")

    for subdir, enc_name in ENCODERS:
        # Load stats for each num_states
        stats_by_ns = {}
        for ns in num_states_list:
            trajs = load_trajs(subdir, ns)
            if trajs is not None:
                stats_by_ns[ns] = compute_stats(trajs, ns)

        for m_idx, (metric_key, metric_label) in enumerate(METRICS):
            if m_idx == 0:
                row = rf"\multirow{{{n_metrics}}}{{*}}{{{enc_name}}}"
            else:
                row = "   "
            row += f" & {metric_label}"

            for ns in num_states_list:
                if ns in stats_by_ns:
                    mean, std = stats_by_ns[ns][metric_key]
                    row += f" & ${fmt_val(metric_key, mean, std)}$"
                else:
                    row += " & --"
            row += r" \\"
            lines.append(row)

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_states", nargs="+", type=int,
                        default=[512, 1024, 1536, 2048])
    parser.add_argument("--save", default=None,
                        help="Save .tex file (default: print to stdout)")
    args = parser.parse_args()

    latex = generate_latex(args.num_states)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            f.write(latex + "\n")
        print(f"Saved {args.save}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
