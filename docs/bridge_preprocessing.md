# Visual Embedding Extraction

Extract visual embeddings from pretrained models (DINOv2, SigLIP) on BridgeData formatted via `data_processing/`.

## Setup

1. Install main project dependencies (from repo root):
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

2. Install visual embedding dependencies:
   ```bash
   pip install -r visual_embedding/requirements.txt
   ```

## Data Format

Uses the same TFRecord format produced by:
1. `data_processing/bridgedata_raw_to_numpy.py`
2. `data_processing/bridgedata_numpy_to_tfrecord.py`

## Supported Models

| Model | Params | Embedding Dim | VRAM (approx) |
|-------|--------|---------------|----------------|
| dinov2_small | 22M | 384 | ~2GB |
| dinov2 | 86M | 768 | ~4GB |
| siglip_base | 86M | 768 | ~4GB |
| dinov2_giant | 1.1B | 1536 | ~8GB |
| siglip_large | 400M | 1024 | ~6GB |

All models run comfortably on an RTX 5060 Ti (16GB).

## Usage

Run from the **project root**:

```bash
# DINOv2 base (recommended for RTX 5060 Ti)
python visual_embedding/extract_pretrained_embeddings.py \
    --data_path /path/to/your/tfrecords \
    --model_name dinov2 \
    --output_dir ./img_features_pretrained \
    --name my_run

# DINOv2 small (faster, less VRAM)
python visual_embedding/extract_pretrained_embeddings.py --model_name dinov2_small

# python visual_embedding/extract_pretrained_embeddings.py \
#     --data_path /path/to/your/tfrecords \
#     --model_name dinov2_small \
#     --output_dir ./img_features_pretrained

# SigLIP base
python visual_embedding/extract_pretrained_embeddings.py \
    --data_path /path/to/your/tfrecords \
    --model_name siglip_base
```

## Full Pipeline (TFDS path, unified — no train/val split)

### Step 1: Extract embeddings from TFDS

All episodes (train+val) processed in one pass with contiguous `traj_idx`.
Also builds `traj_registry.json` inline.

```bash
python visual_embedding/extract_embeddings_tfds.py \
    --data_path /path/to/bridge_dataset/1.0.0 \
    --model_name dinov2_small \
    --output_dir /path/to/viz_features
```

Output: `features.npy`, `actions.npy`, `locations.npy`, `traj_registry.json`

### Step 2: Discretize and build trajectories

```bash
python bridge_model_postprocessing_viz.py \
    --model-name dinov2_small --num-states 512 --num-actions 32
```

Output: `trajs.json`, `traj_idx_order.json`, `trans_probs.npy`, codebooks.

### Step 3: Train model

```bash
python train_bridge.py
```

Cross-validation (KFold) is handled internally by the model.

### Traceability

```
trajs.json[i] -> traj_idx_order[i] -> traj_registry[traj_idx]
     |                                      -> file_path, language, tfds_split
     v
KFold train/test index
```

Fetch raw RGB frame:
```bash
python lookup_frame_tfds.py --traj_json_index 9 --frame_index 4 \
    --traj_idx_order /path/to/traj_idx_order_512_32.json \
    --registry /path/to/traj_registry.json \
    --data_path /path/to/bridge_dataset/1.0.0 --save frame.png
```
