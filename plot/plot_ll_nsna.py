#!/usr/bin/env python3
"""
Visualize test log-likelihood from ll_*.csv of all (ns, na) cases as boxplots.
Handles error cases (missing/invalid csv) so they are visible in the plot.
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


def load_test_ll_for_case(output_case_dir, ll_metric='test_ll', use_largest_num_trajs=True):
    """
    Load all ll_*.csv in output_case_dir and return array of test_ll (or train_ll).
    If use_largest_num_trajs: one value per fold for the largest num_trajs only (5 values).
    Otherwise: all test_ll values from the csv (15 = 3 num_trajs x 5 folds).
    Returns (values_array, error_message). values_array is None on error.
    """
    if not os.path.isdir(output_case_dir):
        return None, "directory missing"
    csvs = [f for f in os.listdir(output_case_dir) if f.startswith('ll_') and f.endswith('.csv')]
    if not csvs:
        return None, "no ll_*.csv"
    all_values = []
    for fname in sorted(csvs):
        path = os.path.join(output_case_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return None, f"read error: {e}"
        if ll_metric not in df.columns:
            return None, f"missing column '{ll_metric}'"
        if use_largest_num_trajs and 'num_trajs' in df.columns:
            max_trajs = df['num_trajs'].max()
            df = df[df['num_trajs'] == max_trajs]
        vals = df[ll_metric].dropna().values
        if len(vals) == 0:
            return None, "no valid values"
        all_values.extend(vals.tolist())
    return np.array(all_values), None


def main():
    parser = argparse.ArgumentParser(description="Boxplot test LL across ns/na cases")
    parser.add_argument('--output_base', type=str,
                        default=os.path.join(REPO_ROOT, 'outputs', 'bridge_autotest'),
                        help='Base dir containing ns_NS_na_NA/ with ll_*.csv')
    parser.add_argument('--data_autotest', type=str, default=DATA_AUTOTEST_DIR,
                        help='Data autotest dir to infer expected cases (for labeling errors)')
    parser.add_argument('--metric', type=str, default='test_ll', choices=['test_ll', 'train_ll'],
                        help='Which column to plot')
    parser.add_argument('--use_all_rows', action='store_true',
                        help='Use all rows per csv; else use largest num_trajs only (one per fold)')
    parser.add_argument('--out', type=str, default=None,
                        help='Save figure path (default: plot/ll_nsna_boxplot.png)')
    parser.add_argument('--title', type=str, default='Test log-likelihood by case (NS/NA)')
    args = parser.parse_args()

    # Determine full case list: expected from data_autotest, then add any from output dirs
    expected = discover_expected_cases(args.data_autotest)
    from_dirs = discover_output_cases(args.output_base)
    if expected:
        cases = expected
        # Include any from output dirs not in expected
        for c in from_dirs:
            if c not in cases:
                cases.append(c)
        cases.sort(key=lambda x: (x[0], x[1]))
    else:
        cases = from_dirs

    if not cases:
        print("No cases found. Check --output_base and --data_autotest.")
        return

    labels = [f"{ns}/{na}" for ns, na in cases]
    data = []
    errors = []  # (index, message)

    for i, (ns, na) in enumerate(cases):
        case_dir = os.path.join(args.output_base, f'ns_{ns}_na_{na}')
        vals, err = load_test_ll_for_case(
            case_dir,
            ll_metric=args.metric,
            use_largest_num_trajs=not args.use_all_rows,
        )
        if err:
            data.append(np.array([np.nan]))  # placeholder so x position is reserved
            errors.append((i, err))
        else:
            data.append(vals)

    error_indices = {e[0] for e in errors}

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(cases) * 0.5), 5))
    positions = np.arange(len(cases))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    # Color boxes: normal vs error
    for i, box in enumerate(bp['boxes']):
        if i in error_indices:
            box.set_facecolor('lightcoral')
            box.set_alpha(0.7)
        else:
            box.set_facecolor('lightsteelblue')
            box.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(args.metric.replace('_', ' ').title())
    ax.set_xlabel('Case (num_states / num_actions)')
    ax.set_title(args.title)

    # Mark error cases on x-axis
    new_labels = []
    for i, lb in enumerate(labels):
        if i in error_indices:
            msg = next(e[1] for e in errors if e[0] == i)
            short = (msg[:18] + '…') if len(msg) > 18 else msg
            new_labels.append(f"{lb}\n(FAIL: {short})")
        else:
            new_labels.append(lb)
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    for i in error_indices:
        ax.get_xticklabels()[i].set_color('red')
        ax.get_xticklabels()[i].set_fontweight('bold')

    # Y range: hide NaN so they don't stretch the axis
    ax.autoscale(axis='y')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.axhline(ymin, color='gray', linewidth=0.5, alpha=0.5)

    # Legend for error vs success
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightsteelblue', alpha=0.8, label='Success'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Error (no/invalid csv)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    out_path = args.out or os.path.join(REPO_ROOT, 'plot', 'll_nsna_boxplot.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if errors:
        print(f"Error cases ({len(errors)}):")
        for i, msg in errors:
            print(f"  {labels[i]}: {msg}")
    plt.close()


if __name__ == '__main__':
    main()
