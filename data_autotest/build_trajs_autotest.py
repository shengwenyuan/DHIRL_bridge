"""
Autotest variant: merge train_trajs_NS_NA.json and val_trajs_NS_NA.json
into trajs_NS_NA.json. Use with --num_states and --num_actions.
All paths are under data_autotest/ (same dir as this script).
"""
import numpy as np
import json
import os
import argparse

root = os.path.abspath(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True, help='NS in train_trajs_NS_NA.json')
    parser.add_argument('--num_actions', type=int, required=True, help='NA in train_trajs_NS_NA.json')
    args = parser.parse_args()
    ns, na = args.num_states, args.num_actions

    train_path = os.path.join(root, f'train_trajs_{ns}_{na}.json')
    val_path = os.path.join(root, f'val_trajs_{ns}_{na}.json')
    out_path = os.path.join(root, f'trajs_{ns}_{na}.json')

    with open(train_path, 'r') as f:
        train_trajs = json.load(f)
    with open(val_path, 'r') as f:
        val_trajs = json.load(f)
    all_trajs = train_trajs + val_trajs

    transformed_trajs = []

    for traj in all_trajs:
        traj_array = np.array(traj)

        states = traj_array[:, 0]
        actions = traj_array[:, 1]
        next_states = np.append(states[1:], 0)
        transformed_traj = np.column_stack([states, actions, next_states])

        transformed_trajs.append(transformed_traj.tolist())

    with open(out_path, 'w') as f:
        json.dump(transformed_trajs, f)
    print(f'Merged trajs written to {out_path}')


if __name__ == '__main__':
    main()
