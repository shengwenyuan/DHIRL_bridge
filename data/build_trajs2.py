import numpy as np
import json
import os
root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(root, 'train_trajs.json'), 'r') as f:
    train_trajs = json.load(f)
with open(os.path.join(root, 'val_trajs.json'), 'r') as f:
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

with open(os.path.join(root, 'trajs.json'), 'w') as f:
    json.dump(transformed_trajs, f)