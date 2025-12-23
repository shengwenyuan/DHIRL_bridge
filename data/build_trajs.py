import numpy as np
import json
import os
root = os.path.abspath(os.path.dirname(__file__))

traj_dict = {}
states = np.load(os.path.join(root, 'discrete_states.npy'))
actions = np.load(os.path.join(root, 'discrete_actions.npy'))
locations = np.load(os.path.join(root, 'locations.npy'))

for i in range(len(locations)):
    traj_idx, time_idx = locations[i]
    if traj_idx not in traj_dict:
        traj_dict[traj_idx] = []
    traj_dict[traj_idx].append({
        'time_index': int(time_idx),
        'state': int(states[i].item()),
        'action': int(actions[i].item()),
    })

for traj_idx in traj_dict:
    traj_dict[traj_idx].sort(key=lambda x: x['time_index'])

all_trajs = []
for traj_idx in sorted(traj_dict.keys()):
    traj = [[pair['state'], pair['action']] for pair in traj_dict[traj_idx]]
    all_trajs.append(traj)

with open(os.path.join(root, 'train_trajs.json'), 'w') as f:
    json.dump(all_trajs, f)
print(f"Saved {len(all_trajs)} trajectories to {os.path.join(root, 'train_trajs.json')}")