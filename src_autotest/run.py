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
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_MODULE = 'src_autotest.train_bridge'

PARAM_KEYS = [
    'num_states', 'num_actions', 'data_dir',
    'll_filename', 'output_dir', 'group_id',
    'discount', 'num_repeats', 'num_latents', 'rand_seed',
    'model_type', 'hidden_dim', 'rnn_hidden_dim', 'num_layers', 'dropout', 'nhead', 'lr',
    'reg_type', 'reg_weight',
    'num_epochs', 'loss_threshold', 'max_iterations',
    'save_npy', 'gate_mode', 'fold_idx', 'max_trajs',
    'paired_fold_seeds', 'p0_artifacts',
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
    parser.add_argument('--max_parallel', type=int, default=1,
                        help='Max experiments to run concurrently (default: 1 = sequential)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get('defaults', {})
    groups = cfg.get('groups', {})

    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = args.log_dir or os.path.join('src_autotest', 'logs', timestamp)
    os.makedirs(log_root, exist_ok=True)
    shutil.copy2(args.config, os.path.join(log_root, 'config.yaml'))
    with open(os.path.join(log_root, 'command.txt'), 'w') as fout:
        fout.write(' '.join(sys.argv) + '\n')

    summary_rows = []

    # Collect all jobs
    all_jobs = []
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
            exp = dict(exp)
            eid = exp.pop('id', f'E{idx:02d}')
            params = {**defaults, **exp}
            params['group_id'] = f'{gid}/{eid}'

            if 'output_dir' in params:
                params['output_dir'] = os.path.join(params['output_dir'], timestamp)

            label = label_from_overrides(exp, defaults)
            cmd = build_command(params)
            log_path = os.path.join(group_log_dir, f'{eid}.log')
            tag = f'{gid}/{eid}'
            all_jobs.append((cmd, log_path, gid, label, tag))

    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()
    git_branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--short'], text=True
    ).strip())
    with open(os.path.join(log_root, 'git_commit.txt'), 'w') as fout:
        fout.write(git_commit + '\n')
    run_manifest = {
        'timestamp': timestamp,
        'config': os.path.abspath(args.config),
        'git_commit': git_commit,
        'git_branch': git_branch,
        'git_dirty': git_dirty,
        'jobs': [
            {'tag': tag, 'command': cmd, 'log': os.path.abspath(log_path)}
            for cmd, log_path, _, _, tag in all_jobs
        ],
    }
    with open(os.path.join(log_root, 'run_manifest.json'), 'w') as fout:
        json.dump(run_manifest, fout, indent=2)

    if args.max_parallel <= 1:
        # Sequential (original behavior)
        for cmd, log_path, gid, label, tag in all_jobs:
            print(f'\n  >> [{tag}] {label}')
            print(f'     log: {log_path}')
            status, elapsed = run_one(cmd, log_path, gid, label)
            print(f'     {status}  ({elapsed})')
            summary_rows.append({
                'tag': tag, 'label': label,
                'status': status, 'elapsed': str(elapsed),
            })
    else:
        print(f'\n  Running up to {args.max_parallel} experiments in parallel')
        for cmd, log_path, _, label, tag in all_jobs:
            print(f'  >> [{tag}] {label}  log: {log_path}')
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            future_to_job = {}
            for cmd, log_path, gid, label, tag in all_jobs:
                future = executor.submit(run_one, cmd, log_path, gid, label)
                future_to_job[future] = (tag, label)
            for future in as_completed(future_to_job):
                tag, label = future_to_job[future]
                status, elapsed = future.result()
                print(f'  [{tag}] {status}  ({elapsed})')
                summary_rows.append({
                    'tag': tag, 'label': label,
                    'status': status, 'elapsed': str(elapsed),
                })

    summary_path = os.path.join(log_root, 'summary.txt')
    with open(summary_path, 'w') as sf:
        sf.write(f'Autotest Summary  {timestamp}\n')
        sf.write('=' * 72 + '\n')
        for row in summary_rows:
            sf.write(f"{row['tag']:<12s}  {row['label']:<40s}  "
                     f"{row['status']:<12s}  {row['elapsed']}\n")

    run_manifest['results'] = summary_rows
    run_manifest['finished'] = datetime.datetime.now().isoformat()
    with open(os.path.join(log_root, 'run_manifest.json'), 'w') as fout:
        json.dump(run_manifest, fout, indent=2)

    print(f'\n\nAll done.  Logs: {log_root}')
    print(f'Summary:   {summary_path}')


if __name__ == '__main__':
    main()
