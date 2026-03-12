"""
python build_trans_autotest.py --num_states NS --num_actions NA --subdir SUBDIR
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
    parser.add_argument('--subdir', type=str, required=True, help='Subdirectory containing trajectory files')
    args = parser.parse_args()
    num_states = args.num_states
    num_actions = args.num_actions

    trajs_path = os.path.join(root, args.subdir, f'trajs_{num_states}_{num_actions}.json')
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

    out_path = os.path.join(root, args.subdir, f'trans_probs_{num_states}_{num_actions}.npy')
    np.save(out_path, transition_probs)

    print(f"Transition probability matrix shape: {transition_probs.shape}")
    print(f"Total (state, action) pairs with observed transitions: {counts.sum()}")
    print(f"Sum of probabilities (should be 0 or 1): {transition_probs[0, 0, :].sum()}")


if __name__ == '__main__':
    main()
