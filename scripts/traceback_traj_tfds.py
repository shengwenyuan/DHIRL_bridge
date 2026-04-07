"""
Trace model output trajectories back to their source in the TFDS Bridge dataset.

Default mode prints a traceback table (registry metadata per trajectory).
With --extract, pulls 256x256 RGB frames from TFDS and saves them as PNGs
for human visual inspection.

Usage:
    # Print traceback table for fold 0, test split:
    python tools/traceback_traj.py \
        --model_output src_autotest/outputs/20260315_081859/G03/E03 \
        --fold 0 --split test

    # Extract frames for specific trajectories:
    python tools/traceback_traj.py \
        --model_output src_autotest/outputs/20260315_081859/G03/E03 \
        --fold 0 --split test --traj_indices 0 5 10 --extract \
        --data_path /path/to/bridge_dataset/1.0.0
"""

import argparse
import json
import os
import sys

# ── Default paths (edit these to match your setup) ──────────────────────────
DEFAULT_MODEL_OUTPUT = "src_autotest/outputs_freshdata/"
DEFAULT_TRAJ_IDX_ORDER = "../discretes/trajs_viz/dinov2_small/traj_idx_order_512_32.json"
DEFAULT_REGISTRY = "../viz_features/pretrained_embeddings/dinov2_small/traj_registry.json"
DEFAULT_DATA_PATH = "../raw_data/tfds/bridge_dataset/1.0.0/"  # TFDS dataset root
# ────────────────────────────────────────────────────────────────────────────
MAX_EXTRACT_TRAJS = 3  # Safety limit to prevent accidentally extracting too many frames

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def find_fold_dirs(model_output: str) -> list:
    """Find all fold_* subdirectories, searching one level of num_trajs subdirs."""
    folds = []
    for name in sorted(os.listdir(model_output)):
        if name.startswith("fold_"):
            folds.append(os.path.join(model_output, name))
    if folds:
        return folds
    for sub in sorted(os.listdir(model_output)):
        sub_path = os.path.join(model_output, sub)
        if os.path.isdir(sub_path):
            for name in sorted(os.listdir(sub_path)):
                if name.startswith("fold_"):
                    folds.append(os.path.join(sub_path, name))
            if folds:
                return folds
    return folds


def _sanitize(text: str, max_len: int = 40) -> str:
    """Make a string safe for use as a directory name."""
    return text.replace("/", "_").replace(" ", "_")[:max_len]


def process_fold(fold_dir, split, traj_idx_order, registry, traj_indices=None,
                 extract=False, data_path=None, output_root=None):
    """Process a single fold directory for the given split."""
    fold_name = os.path.basename(fold_dir)
    idxes_file = os.path.join(fold_dir, f"{split}_idxes.json")

    if not os.path.exists(idxes_file):
        print(f"  SKIP {fold_name}/{split}: {idxes_file} not found", file=sys.stderr)
        return []

    fold_idxes = load_json(idxes_file)

    if traj_indices is not None:
        positions = traj_indices
        for pos in positions:
            if pos < 0 or pos >= len(fold_idxes):
                print(f"  WARNING: traj_index {pos} out of range "
                      f"(fold has {len(fold_idxes)} {split} trajectories)",
                      file=sys.stderr)
    else:
        positions = list(range(len(fold_idxes)))

    results = []
    for fold_i in positions:
        if fold_i >= len(fold_idxes):
            continue
        trajs_json_idx = fold_idxes[fold_i]

        if trajs_json_idx >= len(traj_idx_order):
            print(f"  WARNING: trajs_json_idx {trajs_json_idx} out of range "
                  f"in traj_idx_order (len={len(traj_idx_order)})", file=sys.stderr)
            continue

        traj_idx = traj_idx_order[trajs_json_idx]
        entry = registry.get(str(traj_idx))
        if entry is None:
            print(f"  WARNING: traj_idx {traj_idx} not in registry", file=sys.stderr)
            continue

        results.append({
            "fold_i": fold_i,
            "trajs_json_idx": trajs_json_idx,
            "traj_idx": traj_idx,
            "language": entry.get("language", ""),
            "file_path": entry.get("file_path", ""),
            "tfds_split": entry.get("tfds_split", entry.get("split", "")),
            "traj_length": entry.get("traj_length", 0),
        })

    # Print table
    print(f"\n{fold_name} | {split} | {len(results)}/{len(fold_idxes)} trajectories")
    print(f"  {'fold_i':>6}  {'trajs_idx':>9}  {'traj_idx':>8}  {'len':>4}  "
          f"{'language':<40}  file_path")
    print(f"  {'------':>6}  {'---------':>9}  {'--------':>8}  {'---':>4}  "
          f"{'--------':<40}  ---------")
    for r in results:
        lang_trunc = r["language"][:40]
        fp_trunc = r["file_path"][-60:] if len(r["file_path"]) > 60 else r["file_path"]
        print(f"  {r['fold_i']:>6}  {r['trajs_json_idx']:>9}  {r['traj_idx']:>8}  "
              f"{r['traj_length']:>4}  {lang_trunc:<40}  ...{fp_trunc}")

    # Extract frames from TFDS if requested
    if extract:
        if not data_path:
            print("  ERROR: --data_path required for --extract", file=sys.stderr)
            return results
        if len(results) > MAX_EXTRACT_TRAJS:
            print(f"  ERROR: refusing to extract {len(results)} trajectories "
                  f"(max {MAX_EXTRACT_TRAJS}). Use --traj_indices to narrow down.",
                  file=sys.stderr)
            return results

        # Lazy import — only load TF/TFDS when actually extracting
        from lookup_frame_tfds import _load_episode
        from PIL import Image

        extract_dir = os.path.join(output_root, f"{fold_name}_{split}")
        os.makedirs(extract_dir, exist_ok=True)

        for r in results:
            traj_dir_name = (f"traj_{r['fold_i']:05d}_idx{r['traj_idx']}_"
                             f"{_sanitize(r['language'])}")
            traj_dir = os.path.join(extract_dir, traj_dir_name)
            os.makedirs(traj_dir, exist_ok=True)

            print(f"  Extracting traj fold_i={r['fold_i']} "
                  f"(traj_idx={r['traj_idx']}, {r['traj_length']} frames) ...")

            episode = _load_episode(data_path, r["tfds_split"], r["traj_idx"])
            steps = list(episode["steps"])
            for frame_i, step in enumerate(steps):
                img = step["observation"]["image_0"].numpy()
                Image.fromarray(img).save(
                    os.path.join(traj_dir, f"frame_{frame_i:03d}.png"))

            # Save metadata alongside frames
            meta = {k: r[k] for k in
                    ("fold_i", "trajs_json_idx", "traj_idx", "language",
                     "file_path", "tfds_split", "traj_length")}
            with open(os.path.join(traj_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    -> {len(steps)} frames saved to {traj_dir}/")

        print(f"  All extractions saved under {extract_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Trace model output trajectories back to TFDS Bridge dataset."
    )
    parser.add_argument("--model_output", type=str, default=DEFAULT_MODEL_OUTPUT,
                        help="Path to model output dir containing fold_*/ subdirs.")
    parser.add_argument("--traj_idx_order", type=str, default=DEFAULT_TRAJ_IDX_ORDER,
                        help="Path to traj_idx_order_{NS}_{NA}.json.")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY,
                        help="Path to traj_registry.json.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to TFDS dataset (e.g. .../bridge_dataset/1.0.0). "
                             "Required when --extract is used.")

    parser.add_argument("--fold", type=str, default="0",
                        help="Fold number (0-4) or 'all'.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train", "both"],
                        help="Which split to trace. Default: test.")
    parser.add_argument("--traj_indices", type=int, nargs="+", default=None,
                        help="Specific 0-based trajectory indices within the fold. "
                             "If omitted, process all (table only) or error (extract).")
    parser.add_argument("--extract", action="store_true",
                        help="Extract 256x256 RGB frames from TFDS as PNGs under "
                             "{model_output}/traceback_output/.")

    args = parser.parse_args()

    if not args.registry:
        parser.error("--registry is required (path to traj_registry.json)")
    if args.extract and not args.data_path:
        parser.error("--data_path is required when using --extract")

    # Load shared artifacts
    print(f"Loading traj_idx_order: {args.traj_idx_order}")
    traj_idx_order = load_json(args.traj_idx_order)
    print(f"Loading registry: {args.registry}")
    registry = load_json(args.registry)

    # Resolve fold directories
    fold_dirs = find_fold_dirs(args.model_output)
    if not fold_dirs:
        print(f"ERROR: No fold_* directories found under {args.model_output}",
              file=sys.stderr)
        sys.exit(1)

    if args.fold == "all":
        target_folds = fold_dirs
    else:
        fold_num = int(args.fold)
        target_folds = [d for d in fold_dirs if d.endswith(f"fold_{fold_num}")]
        if not target_folds:
            print(f"ERROR: fold_{fold_num} not found. Available: "
                  f"{[os.path.basename(d) for d in fold_dirs]}", file=sys.stderr)
            sys.exit(1)

    splits = ["test", "train"] if args.split == "both" else [args.split]
    output_root = os.path.join(args.model_output, "traceback_output")

    for fold_dir in target_folds:
        for split in splits:
            process_fold(
                fold_dir=fold_dir,
                split=split,
                traj_idx_order=traj_idx_order,
                registry=registry,
                traj_indices=args.traj_indices,
                extract=args.extract,
                data_path=args.data_path,
                output_root=output_root,
            )


if __name__ == "__main__":
    main()
