# BridgeWAM Ablations

This package contains four ablation models. It does not change the main BridgeWAM
forward path in the parent directory. The `ablation_bridgewam` name preserves the
existing experiment namespace.

| Task | Action routing | Trainable components |
|---|---|---|
| `libero_bridgewam_ablation_frozen_video_2layer_1e-4` | Cross-attention reads 32 LBQs; self-attention reads Action | All ActionDiT parameters and LBQs; Video/Proprio/VAE/Text are frozen |
| `libero_bridgewam_ablation_lbq_kv_mot_2layer_1e-4` | Cross-attention reads Text+State; self-attention reads Action K/V and LBQ K/V | Video, Action, LBQs and Proprio |
| `libero_bridgewam_ablation_idm_2layer_1e-4` | Full future Video reaches Action cross-attention through 32 LBQs | Video, Action, LBQs and Proprio |
| `libero_bridgewam_ablation_joint_30layer_1e-4` | Synchronous Video/LBQ/Action mixed attention at every layer; Action cross-attention reads Text+State | Video, Action, LBQs and Proprio |

All four use 32 LBQs and freeze the VAE and text encoder. The separate cached
Video K/V routing option is disabled; the Joint variant still permits cross-stream
access in its joint attention mask. Checkpoints carry a strict
`bridgewam_ablation` identity and reject loading with the wrong task.

## Frozen Video initialization

The Frozen Video task defaults to the release checkpoint at its original server
path. Override that path with:

```bash
export BRIDGEWAM_RELEASE_CKPT=/path/to/libero_uncond_2cam224.pt
```

Only Video DiT and `proprio_encoder` load from that release checkpoint. ActionDiT
is initialized from `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`, with its
two layers mapped to original layers 0 and 29. The release checkpoint does not
contain the VAE; the VAE loads from the same Wan weights as the baseline and stays
frozen. The legacy `FASTWAM_RELEASE_CKPT` fallback is retained.

## Training

Run from the export root after configuring external weights, datasets and outputs
as described in the root README.

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_bridgewam_ablation_frozen_video_2layer_1e-4

bash scripts/train_zero1.sh 8 \
  task=libero_bridgewam_ablation_lbq_kv_mot_2layer_1e-4

bash scripts/train_zero1.sh 8 \
  task=libero_bridgewam_ablation_idm_2layer_1e-4

bash scripts/train_zero1.sh 8 \
  task=libero_bridgewam_ablation_joint_30layer_1e-4
```

## Full LIBERO evaluation

Set `TASK_NAME` to one of the corresponding task names above and use its matching
checkpoint and dataset statistics:

```bash
python experiments/libero/run_libero_manager.py \
  task="$TASK_NAME" \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=6 \
  MULTIRUN.max_tasks_per_gpu=1
```

IDM and Joint `infer_action()` generate and decode the complete future video.
Frozen Video and LBQ-K/V MoT use the standard action-only LIBERO inference entry.

## Verification in the full repository

The root `tests` directory is intentionally excluded from this source export.
Run the following only from the full repository at the pinned commit, using an
environment with the test dependencies installed:

```bash
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m unittest -v \
  tests.test_latent_bridge_queries \
  tests.test_alternating_action_dit \
  tests.test_bridgewam_ablations
```
