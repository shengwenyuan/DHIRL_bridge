"""
Autotest variant: build trans_probs_NS_NA.npy from trajs_NS_NA.json.
Use with --num_states and --num_actions.
All paths are under data_autotest/ (same dir as this script).
"""
import numpy as np
import json
import os
import argparse

root = os.path.abspath(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True, help='NS (number of states)')
    parser.add_argument('--num_actions', type=int, required=True, help='NA (number of actions)')
    args = parser.parse_args()
    num_states = args.num_states
    num_actions = args.num_actions

    trajs_path = os.path.join(root, f'trajs_{num_states}_{num_actions}.json')
    with open(trajs_path) as f:
        trajs = json.load(f)

    counts = np.zeros((num_states, num_actions, num_states), dtype=np.int16)

    for traj in trajs:
        traj_array = np.array(traj)

        for state, action, next_state in traj_array:
            state = int(state)
            action = int(action)
            next_state = int(next_state)
            counts[state, action, next_state] += 1

    transition_probs = np.zeros((num_states, num_actions, num_states), dtype=np.float64)

    for s in range(num_states):
        for a in range(num_actions):
            total_count = counts[s, a, :].sum()
            if total_count > 0:
                transition_probs[s, a, :] = counts[s, a, :] / total_count

    out_path = os.path.join(root, f'trans_probs_{num_states}_{num_actions}.npy')
    np.save(out_path, transition_probs)

    print(f"Transition probability matrix shape: {transition_probs.shape}")
    print(f"Total (state, action) pairs with observed transitions: {counts.sum()}")
    print(f"Sum of probabilities (should be 0 or 1): {transition_probs[0, 0, :].sum()}")


if __name__ == '__main__':
    main()
