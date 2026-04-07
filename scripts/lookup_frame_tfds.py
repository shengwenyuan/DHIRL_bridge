"""
Trace a trajectory index in trajs.json back to its raw RGB frame via TFDS.

Usage as CLI:
    python lookup_frame_tfds.py --traj_json_index 9 --frame_index 4 \
        --traj_idx_order /path/to/traj_idx_order_512_32.json \
        --registry /path/to/traj_registry.json \
        --data_path /path/to/bridge_dataset/1.0.0 \
        --save frame_out.png

Usage as library:
    from lookup_frame_tfds import fetch_rgb_frame, fetch_trajectory_frames
    img = fetch_rgb_frame(traj_json_index=9, frame_index=4, ...)
    imgs = fetch_trajectory_frames(traj_json_index=9, ...)
"""

import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds


def _load_episode(data_path: str, split: str, traj_idx: int):
    """Load a single episode from TFDS by sequential index."""
    builder = tfds.builder_from_directory(data_path)
    ds = builder.as_dataset(split=split)
    episode = next(iter(ds.skip(traj_idx).take(1)))
    return episode


def fetch_rgb_frame(
    traj_json_index: int,
    frame_index: int,
    traj_idx_order_path: str,
    registry_path: str,
    data_path: str,
) -> np.ndarray:
    """Trace trajs.json[traj_json_index], frame_index -> 256x256x3 uint8 RGB array.

    Returns:
        np.ndarray of shape (256, 256, 3), dtype uint8.
    """
    with open(traj_idx_order_path) as f:
        traj_idx = json.load(f)[traj_json_index]

    with open(registry_path) as f:
        entry = json.load(f)[str(traj_idx)]

    if frame_index < 0 or frame_index >= entry["traj_length"]:
        raise IndexError(
            f"frame_index {frame_index} out of range for traj_idx {traj_idx} "
            f"(length {entry['traj_length']})"
        )

    episode = _load_episode(data_path, entry["split"], traj_idx)
    steps = list(episode["steps"])
    return steps[frame_index]["observation"]["image_0"].numpy()


def fetch_trajectory_frames(
    traj_json_index: int,
    traj_idx_order_path: str,
    registry_path: str,
    data_path: str,
) -> dict:
    """Fetch all frames and metadata for a trajectory.

    Returns:
        dict with keys: images (N,256,256,3), actions (N,7), language, file_path, traj_idx
    """
    with open(traj_idx_order_path) as f:
        traj_idx = json.load(f)[traj_json_index]

    with open(registry_path) as f:
        entry = json.load(f)[str(traj_idx)]

    episode = _load_episode(data_path, entry["split"], traj_idx)
    steps = list(episode["steps"])

    return {
        "images": np.stack([s["observation"]["image_0"].numpy() for s in steps]),
        "actions": np.stack([s["action"].numpy() for s in steps]),
        "language": entry["language"],
        "file_path": entry["file_path"],
        "traj_idx": traj_idx,
        "traj_length": len(steps),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Look up a raw RGB frame from a trajs.json trajectory index (TFDS)."
    )
    parser.add_argument("--traj_json_index", type=int, required=True,
                        help="Index into trajs.json.")
    parser.add_argument("--frame_index", type=int, default=None,
                        help="0-based timestep. If omitted, saves all frames.")
    parser.add_argument("--traj_idx_order", type=str, required=True,
                        help="Path to traj_idx_order_{ns}_{na}.json.")
    parser.add_argument("--registry", type=str, required=True,
                        help="Path to traj_registry.json.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to TFDS dataset (e.g. .../bridge_dataset/1.0.0).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save frame as image (e.g. frame.png), or directory for all frames.")
    args = parser.parse_args()

    with open(args.traj_idx_order) as f:
        traj_idx = json.load(f)[args.traj_json_index]
    with open(args.registry) as f:
        entry = json.load(f)[str(traj_idx)]

    print(f"trajs.json[{args.traj_json_index}] -> traj_idx={traj_idx}")
    print(f"  file_path: {entry['file_path']}")
    print(f"  language:  {entry['language']}")
    print(f"  split:     {entry['split']}")
    print(f"  length:    {entry['traj_length']} frames")

    from PIL import Image

    if args.frame_index is not None:
        img = fetch_rgb_frame(
            args.traj_json_index, args.frame_index,
            args.traj_idx_order, args.registry, args.data_path,
        )
        print(f"  frame {args.frame_index} -> shape {img.shape}")
        if args.save:
            Image.fromarray(img).save(args.save)
            print(f"  Saved to {args.save}")
    else:
        traj = fetch_trajectory_frames(
            args.traj_json_index,
            args.traj_idx_order, args.registry, args.data_path,
        )
        print(f"  Fetched {traj['traj_length']} frames")
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            for i, img in enumerate(traj["images"]):
                path = os.path.join(args.save, f"frame_{i:03d}.png")
                Image.fromarray(img).save(path)
            print(f"  Saved {traj['traj_length']} frames to {args.save}/")


if __name__ == "__main__":
    main()
