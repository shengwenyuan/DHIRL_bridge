import numpy as np
import json

with open('data/train_trajs.json') as f:
    train_trajs = json.load(f)
with open('data/val_trajs.json', 'rb') as f:
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

with open('data/trajs.json', 'w') as f:
    json.dump(transformed_trajs, f)