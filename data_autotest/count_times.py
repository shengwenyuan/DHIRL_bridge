"""
Compute the average number of times each state (and action) appears
within a single trajectory, averaged over all trajectories.

Usage:
    python count_times.py --trajs_path PATH_TO_TRAJS.json
"""
import json
import argparse
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajs_path', type=str, required=True,
                        help='Path to the trajectories JSON file')
    args = parser.parse_args()

    with open(args.trajs_path, 'r') as f:
        trajs = json.load(f)

    state_avg_counts = []
    action_avg_counts = []

    for traj in trajs:
        arr = np.array(traj)
        states = arr[:, 0].astype(int)
        actions = arr[:, 1].astype(int)

        s_counter = Counter(states.tolist())
        a_counter = Counter(actions.tolist())

        # mean occurrence count per unique state/action in this traj
        state_avg_counts.append(np.mean(list(s_counter.values())))
        action_avg_counts.append(np.mean(list(a_counter.values())))

    state_result = np.mean(state_avg_counts)
    action_result = np.mean(action_avg_counts)

    print(f'Number of trajectories : {len(trajs)}')
    print(f'Avg traj length        : {np.mean([len(t) for t in trajs]):.2f}')
    print()
    print(f'Avg occurrence of a state  per traj: {state_result:.4f}')
    print(f'Avg occurrence of an action per traj: {action_result:.4f}')


if __name__ == '__main__':
    main()
