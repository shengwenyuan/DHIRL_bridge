#!/usr/bin/env python
"""
Autotest runner — reads a YAML config and launches train_bridge.py
experiments with full logging.

Usage (from DHIRL_bridge root):
    python -m src_autotest.run src_autotest/configs/test_bridge.yaml
    python -m src_autotest.run src_autotest/configs/test_bridge.yaml --groups model_comparison
"""

import os
import sys
import yaml
import subprocess
import datetime
import argparse

TRAIN_MODULE = 'src_autotest.train_bridge'

PARAM_KEYS = [
    'num_states', 'num_actions', 'data_dir',
    'll_filename', 'output_dir', 'group_id',
    'discount', 'num_repeats', 'num_latents', 'rand_seed',
    'model_type', 'hidden_dim', 'rnn_hidden_dim', 'num_layers', 'dropout', 'nhead', 'lr',
    'reg_type', 'reg_weight',
    'num_epochs', 'loss_threshold', 'max_iterations',
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_command(params):
    cmd = [sys.executable, '-m', TRAIN_MODULE]
    for key in PARAM_KEYS:
        if key in params:
            cmd += [f'--{key}', str(params[key])]
    return cmd


def label_from_overrides(exp, defaults):
    """Derive a short human-readable label from the keys that differ from defaults."""
    parts = []
    for k, v in exp.items():
        if k in defaults and defaults[k] == v:
            continue
        parts.append(f'{k}={v}')
    return ', '.join(parts) if parts else '(defaults)'


def run_one(cmd, log_path, group_name, label):
    start = datetime.datetime.now()
    with open(log_path, 'w') as lf:
        lf.write(f'group  : {group_name}\n')
        lf.write(f'label  : {label}\n')
        lf.write(f'command: {" ".join(cmd)}\n')
        lf.write(f'started: {start.isoformat()}\n')
        lf.write('=' * 72 + '\n\n')
        lf.flush()

        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        end = datetime.datetime.now()
        elapsed = end - start
        lf.write('\n' + '=' * 72 + '\n')
        lf.write(f'finished : {end.isoformat()}\n')
        lf.write(f'elapsed  : {elapsed}\n')
        lf.write(f'exit_code: {result.returncode}\n')

    status = 'OK' if result.returncode == 0 else f'FAIL({result.returncode})'
    return status, elapsed


def main():
    parser = argparse.ArgumentParser(description='Autotest runner')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--groups', type=str, nargs='*', default=None,
                        help='Run only the listed groups (default: all)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log directory')
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get('defaults', {})
    groups = cfg.get('groups', {})

    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = args.log_dir or os.path.join('src_autotest', 'logs', timestamp)
    os.makedirs(log_root, exist_ok=True)

    summary_rows = []

    for group_name, group_cfg in groups.items():
        description = group_cfg.get('description', '')
        experiments = group_cfg.get('experiments', [])
        gid = group_cfg.get('id', group_name)

        group_log_dir = os.path.join(log_root, gid)
        os.makedirs(group_log_dir, exist_ok=True)

        print(f'\n{"="*60}')
        print(f'  Group: {gid}  ({len(experiments)} experiments)')
        if description:
            print(f'  {description}')
        print(f'{"="*60}')

        for idx, exp in enumerate(experiments):
            eid = exp.pop('id', f'E{idx:02d}')
            params = {**defaults, **exp}
            params['group_id'] = f'{gid}/{eid}'

            label = label_from_overrides(exp, defaults)
            cmd = build_command(params)
            log_path = os.path.join(group_log_dir, f'{eid}.log')

            print(f'\n  >> [{gid}/{eid}] {label}')
            print(f'     log: {log_path}')

            status, elapsed = run_one(cmd, log_path, gid, label)

            print(f'     {status}  ({elapsed})')
            summary_rows.append({
                'tag': f'{gid}/{eid}',
                'label': label,
                'status': status,
                'elapsed': str(elapsed),
            })

    summary_path = os.path.join(log_root, 'summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write(f'Autotest Summary  {timestamp}\n')
        sf.write('=' * 72 + '\n')
        for row in summary_rows:
            sf.write(f"{row['tag']:<12s}  {row['label']:<40s}  "
                     f"{row['status']:<12s}  {row['elapsed']}\n")

    print(f'\n\nAll done.  Logs: {log_root}')
    print(f'Summary:   {summary_path}')


if __name__ == '__main__':
    main()
