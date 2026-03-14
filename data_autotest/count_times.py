"""
Diagnose whether KMeans discretization granularity is appropriate.

Reports per-trajectory statistics (mean ± std across trajectories):
  1. num_unique_states   — how many distinct clusters visited per traj
  2. state_coverage      — fraction of total clusters visited per traj
  3. avg_state_occurrence — mean revisit count of visited states per traj
  4. std_state_occurrence — std of revisit counts within each traj
                           (high = one dominant state + many singletons)
  5. singleton_ratio      — fraction of visited states seen only once
                           (high = trajectory is "passing through", 
                            discretization too fine)

Same metrics reported for actions.

Usage:
    python count_times.py --trajs_path PATH_TO_TRAJS.json [--num_states 2048] [--num_actions 32]
"""
import json
import argparse
from collections import Counter

import numpy as np


def compute_stats(trajs, num_states, num_actions):
    records = {
        'state': {
            'num_unique': [], 'coverage': [],
            'avg_occ': [], 'std_occ': [], 'singleton_ratio': []
        },
        'action': {
            'num_unique': [], 'coverage': [],
            'avg_occ': [], 'std_occ': [], 'singleton_ratio': []
        }
    }

    for traj in trajs:
        arr = np.array(traj)
        states = arr[:, 0].astype(int)
        actions = arr[:, 1].astype(int)

        for key, values, total in [('state', states, num_states),
                                    ('action', actions, num_actions)]:
            counter = Counter(values.tolist())
            counts = np.array(list(counter.values()))
            n_unique = len(counter)

            records[key]['num_unique'].append(n_unique)
            records[key]['coverage'].append(n_unique / total)
            records[key]['avg_occ'].append(np.mean(counts))
            records[key]['std_occ'].append(np.std(counts))
            records[key]['singleton_ratio'].append(
                np.sum(counts == 1) / n_unique
            )

    return records


def print_stats(records, label, total):
    print(f'\n{"=" * 55}')
    print(f'  {label.upper()} statistics  (total clusters = {total})')
    print(f'{"=" * 55}')
    for metric, desc in [
        ('num_unique',      'Unique clusters per traj'),
        ('coverage',        'Coverage ratio per traj'),
        ('avg_occ',         'Avg revisit count per traj'),
        ('std_occ',         'Std of revisit counts per traj'),
        ('singleton_ratio', 'Singleton ratio per traj'),
    ]:
        vals = np.array(records[label][metric])
        print(f'  {desc:40s}: {vals.mean():.4f} ± {vals.std():.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajs_path', type=str, required=True)
    parser.add_argument('--num_states', type=int, default=None,
                        help='Total number of KMeans clusters for states. '
                             'Auto-detected from data if not provided.')
    parser.add_argument('--num_actions', type=int, default=None,
                        help='Total number of KMeans clusters for actions. '
                             'Auto-detected from data if not provided.')
    args = parser.parse_args()

    with open(args.trajs_path, 'r') as f:
        trajs = json.load(f)

    # Auto-detect if not provided
    all_states = set()
    all_actions = set()
    for traj in trajs:
        arr = np.array(traj)
        all_states.update(arr[:, 0].astype(int).tolist())
        all_actions.update(arr[:, 1].astype(int).tolist())

    num_states = args.num_states or len(all_states)
    num_actions = args.num_actions or len(all_actions)

    traj_lens = [len(t) for t in trajs]

    print(f'Number of trajectories : {len(trajs)}')
    print(f'Avg traj length        : {np.mean(traj_lens):.2f} ± {np.std(traj_lens):.2f}')
    print(f'Observed unique states : {len(all_states)} / {num_states}')
    print(f'Observed unique actions: {len(all_actions)} / {num_actions}')

    records = compute_stats(trajs, num_states, num_actions)
    print_stats(records, 'state', num_states)
    print_stats(records, 'action', num_actions)

    # Interpretation guide
    print(f'\n{"=" * 55}')
    print(f'  INTERPRETATION GUIDE')
    print(f'{"=" * 55}')

    s_cov = np.mean(records['state']['coverage'])
    s_sing = np.mean(records['state']['singleton_ratio'])

    if s_cov < 0.01:
        print('  [STATE] Coverage < 1%: discretization is very fine.')
        print('          Consider fewer KMeans clusters.')
    elif s_cov < 0.05:
        print('  [STATE] Coverage 1-5%: moderately fine discretization.')
    else:
        print('  [STATE] Coverage > 5%: reasonable granularity.')

    if s_sing > 0.7:
        print('  [STATE] Singleton ratio > 70%: most visited states')
        print('          are seen only once — trajectories are "passing')
        print('          through" clusters, not dwelling. This hurts')
        print('          tabular methods like IAVI that need revisits.')
    elif s_sing > 0.4:
        print('  [STATE] Singleton ratio 40-70%: moderate pass-through.')
    else:
        print('  [STATE] Singleton ratio < 40%: good revisit density.')


if __name__ == '__main__':
    main()