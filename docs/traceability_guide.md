# Traceability Guide: Model Output to Raw RGB Frame

How to trace any per-trajectory model output back to the exact raw image frame
in the Bridge dataset.

## Flowchart

```
 MODEL OUTPUT (per fold)                    DISCRETIZATION ARTIFACTS
 ========================                   ========================

 f_train.npy[i]                             trajs_{NS}_{NA}.json
 f_test.npy[i]                              traj_idx_order_{NS}_{NA}.json
 mask_train.npy[i]                          traj_registry.json
 train_idxes.json
 test_idxes.json

                        TRACEABILITY CHAIN
 ======================================================================

 f_test.npy[i]        The i-th entry in f_test corresponds to the
       |               i-th test trajectory in this fold.
       v
 test_idxes.json[i]   Recover the index into trajs.json.
       |               (test_idxes[i] = idx)
       v
 trajs.json[idx]      The discretized (state, action, next_state)
       |               sequence for this trajectory.
       v
 traj_idx_order[idx]  Map trajs.json position to the original
       |               contiguous traj_idx used during embedding
       |               extraction. (traj_idx_order[idx] = traj_idx)
       v
 traj_registry        Look up metadata by string key str(traj_idx).
 [str(traj_idx)]      Returns: file_path, language, split, traj_length
       |
       v
 TFDS episode         Load the episode from bridge_dataset via
       |               split and traj_idx, then index into steps.
       v
 steps[frame_index]   Raw 256x256x3 uint8 RGB image.
 ["observation"]
 ["image_0"]

 ======================================================================
```

## File Locations and Formats

| Artifact | Typical Path | Format |
|---|---|---|
| `f_train.npy` / `f_test.npy` | `outputs/.../fold_K/` | list of arrays, each `(T_i, num_latents)` |
| `mask_train.npy` / `mask_test.npy` | `outputs/.../fold_K/` | array `(N, T_max)`, binary |
| `train_idxes.json` / `test_idxes.json` | `outputs/.../fold_K/` | JSON array of ints (indices into trajs.json) |
| `trajs_{NS}_{NA}.json` | `data/` or `data_fresh/...` | JSON array of trajectories; each traj is list of `[s, a, s']` |
| `traj_idx_order_{NS}_{NA}.json` | `data/` or `data_fresh/...` | JSON array; position i maps to original traj_idx |
| `traj_registry.json` | embedding output dir | JSON dict; key=str(traj_idx), value={file_path, language, split, traj_length} |

## Step-by-Step Example

Goal: find the raw RGB frame at timestep 4 of the 0th test trajectory in fold 2.

```python
import json, numpy as np

# 1. Load fold indices
with open('outputs/.../fold_2/test_idxes.json') as f:
    test_idxes = json.load(f)

trajs_json_idx = test_idxes[0]       # e.g. 137

# 2. Map to original traj_idx
with open('data/traj_idx_order_512_32.json') as f:
    traj_idx = json.load(f)[trajs_json_idx]   # e.g. 4821

# 3. Look up metadata
with open('path/to/traj_registry.json') as f:
    entry = json.load(f)[str(traj_idx)]
# entry = {"file_path": "...", "language": "pick up the sponge", "split": "train", "traj_length": 47}

# 4. Fetch the frame via lookup_frame_tfds.py
from lookup_frame_tfds import fetch_rgb_frame
img = fetch_rgb_frame(
    traj_json_index=trajs_json_idx,
    frame_index=4,
    traj_idx_order_path='data/traj_idx_order_512_32.json',
    registry_path='path/to/traj_registry.json',
    data_path='/path/to/bridge_dataset/1.0.0',
)
# img.shape == (256, 256, 3), dtype uint8
```

Or via CLI:
```bash
python lookup_frame_tfds.py \
    --traj_json_index 137 --frame_index 4 \
    --traj_idx_order data/traj_idx_order_512_32.json \
    --registry /path/to/traj_registry.json \
    --data_path /path/to/bridge_dataset/1.0.0 \
    --save frame.png
```

## Notes

- `train_idxes` and `test_idxes` are indices into the **full** `trajs.json` array (not a subset).
- KFold uses `n_splits=5, shuffle=True, random_state=10042`. The saved JSON files make re-derivation unnecessary.
- `traj_idx_order` may be shorter than the raw episode count because length-1 trajectories are dropped during discretization.
- `r_{k}.npy` and `q_{k}.npy` are indexed by `(state, action)`, not by trajectory -- no traceability mapping needed for those.
