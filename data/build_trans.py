import numpy as np
import json

with open('data/trajs.json') as f:
    trajs = json.load(f)

num_states = 256
num_actions = 8

counts = np.zeros((num_states, num_actions, num_states), dtype=np.int8)

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

np.save('data/trans_probs.npy', transition_probs)

print(f"Transition probability matrix shape: {transition_probs.shape}")
print(f"Total (state, action) pairs with observed transitions: {counts.sum()}")
# print(f"Total (state, action) pairs with observed transitions: {np.sum(counts.sum(axis=2) > 0)}")
print(f"Sample probabilities for state 0, action 0:")
print(f"Sum of probabilities (should be 0 or 1): {transition_probs[0, 0, :].sum()}")