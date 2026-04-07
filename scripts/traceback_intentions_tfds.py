"""
Trace model output trajectories by directly loading f_test.npy from a fold directory.

Computes per-timestep intention class assignments (greedy argmax over K softmax
probabilities), prints a summary table, and optionally extracts 256x256 RGB
frames from TFDS organized by class label for human visual inspection.

Usage:
    # Print class-label table for trajectories 0, 5, 10 in fold_0:
    python scripts/traceback_byf_tfds.py \
        --fold_dir src_autotest/outputs_freshdata/.../fold_0 \
        --traj_indices 0 5 10

    # Extract frames organized by class:
    python scripts/traceback_byf_tfds.py \
        --fold_dir src_autotest/outputs_freshdata/.../fold_0 \
        --traj_indices 0 5 10 --extract
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

# ── Default paths (edit these to match your setup) ──────────────────────────
DEFAULT_TRAJ_IDX_ORDER = "../discretes/trajs_viz/dinov2_small/traj_idx_order_512_32.json"
DEFAULT_REGISTRY = "../viz_features/pretrained_embeddings/dinov2_small/traj_registry.json"
DEFAULT_DATA_PATH = "../raw_data/tfds/bridge_dataset/1.0.0/"
# ────────────────────────────────────────────────────────────────────────────
MAX_EXTRACT_TRAJS = 3


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_fold_artifacts(fold_dir: str):
    """Load f_test, mask_test, and test_idxes from a fold directory."""
    f_test = np.load(os.path.join(fold_dir, "f_test.npy"), allow_pickle=True)
    mask_test = np.load(os.path.join(fold_dir, "mask_test.npy"))
    test_idxes = load_json(os.path.join(fold_dir, "test_idxes.json"))
    K = f_test[0].shape[1]
    print(f"Loaded fold: {fold_dir}")
    print(f"  num_trajs={len(f_test)}, K={K}, T_max={f_test[0].shape[0]}")
    return f_test, mask_test, test_idxes, K


def get_class_labels(f_test, mask_test, idx: int):
    """Compute greedy class assignments for a single trajectory."""
    probs = f_test[idx]                    # (T_max, K)
    real_len = int(mask_test[idx].sum())
    probs_valid = probs[:real_len]         # (real_len, K)
    class_labels = np.argmax(probs_valid, axis=1)  # (real_len,)
    return class_labels, probs_valid, real_len


def resolve_metadata(idx, test_idxes, traj_idx_order, registry):
    """Trace fold index -> trajs.json index -> traj_idx -> registry entry."""
    trajs_json_idx = test_idxes[idx]
    traj_idx = traj_idx_order[trajs_json_idx]
    entry = registry.get(str(traj_idx))
    return trajs_json_idx, traj_idx, entry


def print_traj_summary(idx, class_labels, real_len, K, trajs_json_idx, traj_idx, entry):
    """Print class assignment summary for one trajectory."""
    counts = Counter(int(c) for c in class_labels)
    lang = entry.get("language", "") if entry else "??"

    print(f"\n  [{idx}] traj_idx={traj_idx}  len={real_len}  \"{lang}\"")
    # Compact class label sequence
    label_str = "".join(str(c) for c in class_labels)
    # Wrap at 80 chars
    for start in range(0, len(label_str), 80):
        prefix = "    labels: " if start == 0 else "            "
        print(f"{prefix}{label_str[start:start+80]}")
    # Per-class counts
    count_parts = [f"class_{k}:{counts.get(k, 0)}" for k in range(K)]
    print(f"    counts: {', '.join(count_parts)}")


def extract_traj_frames(idx, class_labels, real_len, traj_idx, entry,
                        data_path, fold_dir):
    """Extract frames from TFDS and save organized by class."""
    from lookup_frame_tfds import _load_episode
    from PIL import Image

    traj_dir = os.path.join(fold_dir, "traceback_output",
                            f"traj_{idx:05d}_idx{traj_idx}")
    os.makedirs(traj_dir, exist_ok=True)

    # Create class subdirs
    K = int(class_labels.max()) + 1
    for k in range(K):
        os.makedirs(os.path.join(traj_dir, f"class_{k}"), exist_ok=True)

    tfds_split = entry.get("tfds_split", entry.get("split", ""))
    print(f"  Extracting [{idx}] traj_idx={traj_idx} ({real_len} frames) ...")
    episode = _load_episode(data_path, tfds_split, traj_idx)
    steps = list(episode["steps"])

    for t in range(real_len):
        img = steps[t]["observation"]["image_0"].numpy()
        k = int(class_labels[t])
        Image.fromarray(img).save(
            os.path.join(traj_dir, f"class_{k}", f"frame_{t:03d}.png"))

    # Save summary
    summary = {
        "fold_index": idx,
        "traj_idx": traj_idx,
        "language": entry.get("language", ""),
        "file_path": entry.get("file_path", ""),
        "tfds_split": tfds_split,
        "real_len": real_len,
        "class_labels": [int(c) for c in class_labels],
    }
    with open(os.path.join(traj_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    -> saved to {traj_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Trace fold trajectories via f_test.npy with class labels."
    )
    parser.add_argument("--fold_dir", type=str, required=True,
                        help="Path to fold directory (contains f_test.npy, "
                             "mask_test.npy, test_idxes.json).")
    parser.add_argument("--traj_idx_order", type=str, default=DEFAULT_TRAJ_IDX_ORDER,
                        help="Path to traj_idx_order_{NS}_{NA}.json.")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY,
                        help="Path to traj_registry.json.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to TFDS dataset. Required when --extract is used.")
    parser.add_argument("--traj_indices", type=int, nargs="+", required=True,
                        help="0-based trajectory indices within the fold's test set.")
    parser.add_argument("--extract", action="store_true",
                        help="Extract 256x256 RGB frames from TFDS, organized by class.")

    args = parser.parse_args()

    if args.extract and not args.data_path:
        parser.error("--data_path is required when using --extract")
    if args.extract and len(args.traj_indices) > MAX_EXTRACT_TRAJS:
        parser.error(f"Cannot extract more than {MAX_EXTRACT_TRAJS} trajectories at once. "
                     f"Got {len(args.traj_indices)}.")

    # Load artifacts
    f_test, mask_test, test_idxes, K = load_fold_artifacts(args.fold_dir)
    traj_idx_order = load_json(args.traj_idx_order)
    registry = load_json(args.registry)

    # Validate indices
    for idx in args.traj_indices:
        if idx < 0 or idx >= len(f_test):
            print(f"ERROR: traj_index {idx} out of range (fold has {len(f_test)} "
                  f"test trajectories)", file=sys.stderr)
            sys.exit(1)

    # Process each trajectory
    print(f"\nK={K} intention classes | {len(args.traj_indices)} trajectories selected")
    for idx in args.traj_indices:
        class_labels, probs_valid, real_len = get_class_labels(f_test, mask_test, idx)
        trajs_json_idx, traj_idx, entry = resolve_metadata(
            idx, test_idxes, traj_idx_order, registry)
        print_traj_summary(idx, class_labels, real_len, K,
                           trajs_json_idx, traj_idx, entry)

        if args.extract:
            if entry is None:
                print(f"  SKIP extraction: traj_idx {traj_idx} not in registry",
                      file=sys.stderr)
                continue
            extract_traj_frames(idx, class_labels, real_len, traj_idx, entry,
                                args.data_path, args.fold_dir)


if __name__ == "__main__":
    main()
