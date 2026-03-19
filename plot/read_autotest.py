"""
read_autotest.py — load and summarise autotest results for a given config + result dir.

Typical use (as a module imported by plotting scripts):

    from plot.read_autotest import load_results
    rows = load_results('src_autotest/configs/test_0314.yaml',
                        'src_autotest/outputs/20260315_081859')

Standalone (prints a table):

    python plot/read_autotest.py src_autotest/configs/test_0314.yaml \\
                                 src_autotest/outputs/20260315_081859
"""

import os
import csv
import yaml
import argparse
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _label_from_overrides(exp_overrides, defaults):
    parts = []
    for k, v in exp_overrides.items():
        if k in defaults and defaults[k] == v:
            continue
        # shorten common data_dir prefix for readability
        if k == 'data_dir':
            v = os.path.basename(str(v))
        parts.append(f'{k}={v}')
    return ', '.join(parts) if parts else '(defaults)'


def _read_test_ll(csv_path):
    rows_by_nt = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('test_ll'):
                continue
            nt = float(row['num_trajs'])
            rows_by_nt[nt].append(float(row['test_ll']))

    if not rows_by_nt:
        return None, []

    max_nt = max(rows_by_nt)
    return int(max_nt), rows_by_nt[max_nt][:5]   # cap at 5 folds


# ── public API ────────────────────────────────────────────────────────────────

def load_results(config_path, result_dir, groups=None):
    """
    Parse *config_path* and read ll.csv results from *result_dir*.

    Parameters
    ----------
    config_path : str
        Path to a src_autotest/configs/*.yaml file.
    result_dir  : str
        Path to the timestamped output root, e.g.
        'src_autotest/outputs/20260315_081859'.
    groups      : list[str] | None
        If given, only include groups whose YAML key is in this list.

    """
    cfg = _load_config(config_path)
    defaults = cfg.get('defaults', {})
    all_groups = cfg.get('groups', {})

    if groups:
        all_groups = {k: v for k, v in all_groups.items() if k in groups}

    records = []
    for group_key, group_cfg in all_groups.items():
        gid = group_cfg.get('id', group_key)
        experiments = group_cfg.get('experiments', [])

        for idx, exp in enumerate(experiments):
            exp = dict(exp)                          # don't mutate config
            eid = exp.pop('id', f'E{idx:02d}')
            overrides = dict(exp)
            params = {**defaults, **exp}

            label = _label_from_overrides(overrides, defaults)

            csv_path = os.path.join(result_dir, gid, eid,
                                    params.get('ll_filename', 'll.csv'))
            csv_path = os.path.abspath(csv_path)

            if os.path.isfile(csv_path):
                num_trajs, values = _read_test_ll(csv_path)
            else:
                num_trajs, values = None, []

            if len(values) >= 2:
                import statistics
                mean = statistics.mean(values)
                std  = statistics.stdev(values)
            elif len(values) == 1:
                mean = values[0]
                std  = 0.0
            else:
                mean = std = None

            records.append(dict(
                tag=f'{gid}/{eid}',
                group_key=group_key,
                group_id=gid,
                exp_id=eid,
                label=label,
                overrides=overrides,
                params=params,
                num_trajs=num_trajs,
                test_ll=values,
                mean=mean,
                std=std,
                csv_path=csv_path if os.path.isfile(csv_path) else None,
            ))

    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_table(records):
    tag_w   = max(len(r['tag'])   for r in records)
    label_w = max(len(r['label']) for r in records)

    header = (f"{'tag':<{tag_w}}  {'label':<{label_w}}  "
              f"{'n_trajs':>8}  {'mean_test_ll':>13}  {'std':>8}  folds")
    print(header)
    print('-' * len(header))

    for r in records:
        nt   = str(r['num_trajs']) if r['num_trajs'] else 'N/A'
        mn   = f"{r['mean']:+.4f}" if r['mean']  is not None else '   N/A'
        sd   = f"{r['std']:.4f}"   if r['std']   is not None else '  N/A'
        vals = '  '.join(f"{v:+.4f}" for v in r['test_ll'])
        if not vals:
            vals = 'MISSING'
        print(f"{r['tag']:<{tag_w}}  {r['label']:<{label_w}}  "
              f"{nt:>8}  {mn:>13}  {sd:>8}  [{vals}]")


def main():
    parser = argparse.ArgumentParser(description='Print autotest test_ll summary')
    parser.add_argument('config',     help='Path to YAML config')
    parser.add_argument('result_dir', help='Path to timestamped output root')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Restrict to these YAML group keys')
    args = parser.parse_args()

    records = load_results(args.config, args.result_dir, groups=args.groups)
    if not records:
        print('No experiments found.')
        return

    # group output by group_id for readability
    current_gid = None
    by_group = {}
    for r in records:
        by_group.setdefault(r['group_id'], []).append(r)

    for gid, rows in by_group.items():
        group_key = rows[0]['group_key']
        print(f'\n[{gid}]  {group_key}')
        _print_table(rows)


if __name__ == '__main__':
    main()
