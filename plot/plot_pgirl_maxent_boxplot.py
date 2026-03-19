"""
Boxplot comparison of PGIQL (pgirl) vs MaxEnt IRL test log-likelihoods across ns/na cases.
Reads ll_pgiql.csv from bridge_autotest and ll_max_entropy.csv from maxent_autotest.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_AUTOTEST_DIR = os.path.join(REPO_ROOT, 'data_autotest')


def discover_expected_cases(data_autotest_dir):
    """Expected (ns, na) from data_autotest train_trajs_NS_NA.json / val_trajs_NS_NA.json."""
    if not os.path.isdir(data_autotest_dir):
        return []
    pattern = re.compile(r'^train_trajs_(\d+)_(\d+)\.json$')
    cases = []
    for name in os.listdir(data_autotest_dir):
        m = pattern.match(name)
        if not m:
            continue
        ns, na = int(m.group(1)), int(m.group(2))
        val_name = f'val_trajs_{ns}_{na}.json'
        if os.path.isfile(os.path.join(data_autotest_dir, val_name)):
            cases.append((ns, na))
    cases.sort(key=lambda x: (x[0], x[1]))
    return cases


def discover_output_cases(output_base):
    """(ns, na) from existing output dirs ns_NS_na_NA under output_base."""
    if not os.path.isdir(output_base):
        return []
    pattern = re.compile(r'^ns_(\d+)_na_(\d+)$')
    cases = []
    for name in os.listdir(output_base):
        m = pattern.match(name)
        if not m:
            continue
        path = os.path.join(output_base, name)
        if os.path.isdir(path):
            cases.append((int(m.group(1)), int(m.group(2))))
    cases.sort(key=lambda x: (x[0], x[1]))
    return cases


def load_test_ll_from_csv(case_dir, csv_name, ll_metric='test_ll'):
    """Load test_ll (or train_ll) values from a CSV in case_dir. Returns (values, error_msg)."""
    if not os.path.isdir(case_dir):
        return None, "directory missing"
    csv_path = os.path.join(case_dir, csv_name)
    if not os.path.isfile(csv_path):
        return None, f"missing {csv_name}"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"read error: {e}"
    if ll_metric not in df.columns:
        return None, f"missing column '{ll_metric}'"
    vals = df[ll_metric].dropna().values
    if len(vals) == 0:
        return None, "no valid values"
    return np.array(vals, dtype=float), None


def main():
    parser = argparse.ArgumentParser(
        description="Boxplot comparison: PGIQL vs MaxEnt test LL across ns/na cases"
    )
    parser.add_argument('--pgirl_base', type=str,
                        default=os.path.join(REPO_ROOT, 'outputs', 'bridge_autotest'),
                        help='Base dir for PGIQL outputs (ns_NS_na_NA/ll_pgiql.csv)')
    parser.add_argument('--maxent_base', type=str,
                        default=os.path.join(REPO_ROOT, 'outputs', 'maxent_autotest'),
                        help='Base dir for MaxEnt outputs (ns_NS_na_NA/ll_max_entropy.csv)')
    parser.add_argument('--pgirl_csv', type=str, default='ll_pgiql.csv',
                        help='CSV filename under each PGIQL case dir')
    parser.add_argument('--maxent_csv', type=str, default='ll_max_entropy.csv',
                        help='CSV filename under each MaxEnt case dir')
    parser.add_argument('--data_autotest', type=str, default=DATA_AUTOTEST_DIR,
                        help='Data autotest dir to infer expected cases')
    parser.add_argument('--metric', type=str, default='test_ll', choices=['test_ll', 'train_ll'],
                        help='Which column to plot')
    parser.add_argument('--out', type=str, default=None,
                        help='Save figure path (default: plot/pgirl_maxent_boxplot.png)')
    parser.add_argument('--title', type=str, default='PGIQL vs MaxEnt: test log-likelihood by case')
    parser.add_argument('--ns', type=int, nargs='*', default=None,
                        help='Optional list of num_states to include (e.g. --ns 480 512)')
    parser.add_argument('--na', type=int, nargs='*', default=None,
                        help='Optional list of num_actions to include (e.g. --na 32)')
    parser.add_argument('--neg_ll', action='store_true',
                        help='Plot negative log-likelihood (lower is better)')
    args = parser.parse_args()

    # Cases: union of expected and those present in either output base
    expected = discover_expected_cases(args.data_autotest)
    pgirl_cases = discover_output_cases(args.pgirl_base)
    maxent_cases = discover_output_cases(args.maxent_base)
    all_cases = set(expected) | set(pgirl_cases) | set(maxent_cases)
    cases = sorted(all_cases, key=lambda x: (x[0], x[1]))

    if args.ns is not None:
        cases = [c for c in cases if c[0] in args.ns]
    if args.na is not None:
        cases = [c for c in cases if c[1] in args.na]

    if not cases:
        print("No cases found. Check --pgirl_base / --maxent_base / --data_autotest / --ns / --na.")
        return

    labels = [f"{ns}/{na}" for ns, na in cases]
    data_pgirl = []
    data_maxent = []
    errors_pgirl = []
    errors_maxent = []

    for i, (ns, na) in enumerate(cases):
        pgirl_dir = os.path.join(args.pgirl_base, f'ns_{ns}_na_{na}')
        maxent_dir = os.path.join(args.maxent_base, f'ns_{ns}_na_{na}')

        vals_p, err_p = load_test_ll_from_csv(pgirl_dir, args.pgirl_csv, args.metric)
        vals_m, err_m = load_test_ll_from_csv(maxent_dir, args.maxent_csv, args.metric)

        if err_p:
            data_pgirl.append(np.array([np.nan]))
            errors_pgirl.append((i, err_p))
        else:
            if args.neg_ll:
                vals_p = -vals_p
            data_pgirl.append(vals_p)

        if err_m:
            data_maxent.append(np.array([np.nan]))
            errors_maxent.append((i, err_m))
        else:
            if args.neg_ll:
                vals_m = -vals_m
            data_maxent.append(vals_m)

    # Plot: grouped boxplots (two boxes per case); larger boxes, bold mean lines
    n_cases = len(cases)
    width = 0.52
    positions_pgirl = np.arange(n_cases) - width / 2
    positions_maxent = np.arange(n_cases) + width / 2

    # Mean line colors aligned with box colors (bright blue / bright orange)
    mean_color_pgirl = '#2196F3'
    mean_color_maxent = '#FF9800'

    fig, ax = plt.subplots(figsize=(max(8, n_cases * 0.9), 6))

    bp_pgirl = ax.boxplot(
        data_pgirl,
        positions=positions_pgirl,
        widths=width,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=mean_color_pgirl, linewidth=2.2, linestyle='-', zorder=5),
        medianprops=dict(linewidth=0.8, color='gray', alpha=0.7),
    )
    bp_maxent = ax.boxplot(
        data_maxent,
        positions=positions_maxent,
        widths=width,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=mean_color_maxent, linewidth=2.2, linestyle='-', zorder=5),
        medianprops=dict(linewidth=0.8, color='gray', alpha=0.7),
    )

    for box in bp_pgirl['boxes']:
        box.set_facecolor('steelblue')
        box.set_alpha(0.85)
        box.set_linewidth(1.2)
    for box in bp_maxent['boxes']:
        box.set_facecolor('darkorange')
        box.set_alpha(0.85)
        box.set_linewidth(1.2)

    ax.set_xticks(np.arange(n_cases))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ylabel = 'Negative log-likelihood' if args.neg_ll else 'Test log-likelihood'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Case (num_states / num_actions)')
    ax.set_title(args.title)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.85, label='PGIQL'),
        Patch(facecolor='darkorange', alpha=0.85, label='MaxEnt'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.autoscale(axis='y')
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    out_path = args.out or os.path.join(REPO_ROOT, 'plot', 'pgirl_maxent_boxplot.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    if errors_pgirl or errors_maxent:
        if errors_pgirl:
            print("PGIQL errors:")
            for i, msg in errors_pgirl:
                print(f"  {labels[i]}: {msg}")
        if errors_maxent:
            print("MaxEnt errors:")
            for i, msg in errors_maxent:
                print(f"  {labels[i]}: {msg}")
    plt.close()


if __name__ == '__main__':
    main()
