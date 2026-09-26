# BridgeWAM

## Source-only export

This export is pinned to commit `9761d9e684b9f5ac2f44fd9735395c15af563fa5`.
All model, configuration, training and evaluation program files are byte-for-byte
identical to that commit. Export changes are limited to README documentation,
`.gitignore`, and the legacy RoboTwin policy symlink, which is now relative.
The original repository is not modified.

The included paths are listed below. Git history, the root test suite, bundled
LIBERO / LIBERO-Plus / LIBERO-Pro environments, other third-party trees, datasets,
checkpoints, logs, evaluation results, caches and paper/visualization material are
excluded. Training and simulation evaluation entry points are retained, including
the LIBERO-Pro evaluator added in this version. The historical `README_zh.md`
filename is retained as an identical English copy of this README.

The `src/bridgewam` package owns the implementation. `src/fastwam/__init__.py`
provides lazy legacy imports, including the historical `aclation_bridgewam`
spelling. Embedded model and dataset components under `src` are retained because
they are part of the verified pipeline. See [Checkpoint Compatibility](#checkpoint-compatibility)
for naming compatibility and the explicit FastWAM baseline fallback.

**RoboTwin snapshot limitation:** the pinned commit contains 1,037 zero-byte
regular files under `third_party/RoboTwin`, plus one legacy policy symlink.
This export preserves those files and repairs the symlink. They are placeholders,
not a runnable RoboTwin simulator. Restore matching simulator code and assets
from your verified environment before running RoboTwin evaluation.

Export integrity checks establish source identity; they do not rerun GPU training,
restore full training checkpoints or measure simulator success rates.
`scripts/verify_bridgewam_migration.py` requires Git history and must be run from
the original full repository. Test commands mentioned in the ablation README also
require that repository's `tests` directory.

BridgeWAM training and evaluation code, derived from FastWAM.

Naming, architecture fallback, and checkpoint compatibility: [compatibility notes](#checkpoint-compatibility).

LIBERO-Pro evaluation: [setup, task selection, multi-GPU launcher and result checks](experiments/libero-pro/README.md).
`src/bridgewam` owns the implementation; `src/fastwam/__init__.py` installs lazy legacy
imports, including the historical `aclation_bridgewam` spelling. Both namespaces
resolve to the same modules and classes, preserving existing Hydra targets and checkpoint names.

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[English copy (legacy filename)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

This repository contains BridgeWAM training code and simulation evaluation tools for LIBERO, LIBERO-Plus, LIBERO-Pro and RoboTwin.

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Checkpoint Compatibility](#checkpoint-compatibility)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
BridgeWAM-code/
|-- configs/                 # Model, training and simulation evaluation configs
|-- experiments/             # LIBERO, LIBERO-Plus, LIBERO-Pro and RoboTwin tools
|-- scripts/                 # Training, preprocessing, profiling and utilities
|-- src/                     # bridgewam, fastwam compatibility and LBQ packages
|-- third_party/
|   `-- RoboTwin/            # Pinned placeholders; simulator setup is still needed
|-- .gitignore
|-- LICENSE
|-- README.md
|-- README_zh.md              # Identical English copy; historical filename
|-- __init__.py
`-- pyproject.toml
```

## Environment Setup

```bash
conda create -n bridgewam python=3.10 -y
conda activate bridgewam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

Run commands from the export root. Reuse the dependency versions and simulator
resources from your verified setup; `pyproject.toml` is unchanged and targets a
CUDA training environment. To use the source without relying on another checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

LIBERO environments are external to this export. For standard LIBERO, install the
matching environment and prepend its checkout when needed:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export PYTHONPATH="$PWD/src:$PWD:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Keep its `LIBERO_CONFIG_PATH` configuration pointed at the matching BDDL files,
initial states and simulator assets. LIBERO-Plus and LIBERO-Pro managers create
separate path configurations per run:

- **LIBERO-Plus:** pass `LIBERO_PLUS.repo_path=/path/to/LIBERO-plus` to the manager
  or stage launcher. See the [Plus evaluator guide](experiments/libero-plus/README.md).
- **LIBERO-Pro:** set `LIBERO_PRO_REPO=/path/to/LIBERO-PRO` or pass
  `LIBERO_PRO.repo_path=/path/to/LIBERO-PRO`. See the [Pro evaluator guide](experiments/libero-pro/README.md).
- **RoboTwin:** restore the working simulator at `third_party/RoboTwin`, including
  `task_config/_eval_step_limit.yml`, scripts and assets. Its manager reads that
  local step-limit file even when the worker's `EVALUATION.robotwin_root` is changed.
  The legacy `policy/fastwam_policy` link resolves to the retained compatibility
  adapter; the evaluator also installs the selected `bridgewam_policy` adapter.

No datasets, pretrained weights or simulator downloads are performed by this
export. Existing server paths in configs and launchers are intentionally preserved.
On another machine, set `DIFFSYNTH_MODEL_BASE_PATH` and
`BRIDGEWAM_TRAIN_OUTPUT_BASE`, and override `data.train.dataset_dirs`,
`data.train.text_embedding_cache_dir`, `data.train.pretrained_norm_stats` and
evaluation statistics/checkpoint paths as appropriate.

## Model Preparation

This step is required before both training and inference.

Step 1: set the Wan model directory first (optional, default `./checkpoints`):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: pre-generate the ActionDiT backbone (interpolated from Wan22 DiT):

```bash
# uncond (BridgeWAM package, FastWAM-compatible baseline)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/bridgewam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

The preprocessed LIBERO dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

Download all compressed files first, then extract them all:

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# Run after downloading all 4 tar.gz files
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

The extracted directory structure should be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Download all split archive files first, then concatenate and extract:

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# Run after downloading all robotwin2.0.tar.gz.part-* files
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

The extracted directory structure should be:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

If you also keep:

```text
data/robotwin2.0/dataset_stats.json
```

in the root directory, it can be used directly as the statistics file for the current configs in this repo. You can also recompute it.

## Inference with Released Checkpoints

The released checkpoints and their corresponding dataset stats are available on [Hugging Face](https://huggingface.co/yuanty/fastwam).

Optional: download released checkpoints and dataset stats from Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

After downloading, the local directory is expected to contain:

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

Before running the `LIBERO` benchmark, install the official LIBERO environment first
from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Then run this final step:

```bash
pip install mujoco==3.3.2
```

The `mujoco` environment should ideally stay consistent with the LIBERO data version.

The retained `third_party/RoboTwin` files are placeholders in the pinned commit.
Restore the matching working simulator and follow the official instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) to finish environment installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/bridgewam_policy" "$(pwd)/third_party/RoboTwin/policy/bridgewam_policy"
```

Optional: evaluate released LIBERO checkpoint:

The released `LIBERO` / `RoboTwin` evaluation managers default to `8` GPUs
(`MULTIRUN.num_gpus=8` in `configs/sim_libero.yaml` and `configs/sim_robotwin.yaml`).
If you want to evaluate with fewer GPUs, pass a smaller value such as
`MULTIRUN.num_gpus=4`.

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Optional: evaluate released RoboTwin checkpoint:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

For faster RoboTwin evaluation, we have enabled `EVALUATION.skip_get_obs_within_replan=true` in [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml).
This skips RGB rendering while consecutively executing an action chunk within one replan window, which speeds up evaluation but makes the saved video look very low-FPS.
Set it to `false` if you want to save a fully rendered video.

**Note:** We evaluate with **unseen** instructions, following Motus. [Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308) uses **seen** instructions instead. You can try `EVALUATION.instruction_type=seen` to use **seen** instructions, which should theoretically improve performance by one or two points.

## Training

### 1) Precompute T5 embedding cache before training

Use `scripts/precompute_text_embeds.py` to precompute embeddings for each training task:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### 2) Training the baseline tasks

When running a new task for the first time, set `pretrained_norm_stats` in the corresponding `configs/data/*.yaml` to `null` first.
After one training run, a `dataset_stats.json` file will be generated in the current run directory (for example, `runs/{task_name}/{run_id}/dataset_stats.json`).
You can then update `pretrained_norm_stats` to that file path for subsequent runs.

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

For LIBERO, we train on a single node with 8 GPUs. For RoboTwin, we use 64 GPUs to accelerate training. You can try reducing the GPU count or training epochs.

### 3) Training the main LBQ task

The baseline examples above do not enable the main LBQ architecture. The retained
main experiment uses 32 LBQs, a 30-layer Video DiT and a two-layer alternating
cross/self-attention Action head:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export BRIDGEWAM_TRAIN_OUTPUT_BASE=/path/to/training-results

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_zero1.sh 8 \
    task=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4
```

Prepare the text embedding cache with this same task and set the dataset/statistics
paths before launching. `train_zero1.sh` uses a detached tmux session by default;
set `BRIDGEWAM_TMUX_DISABLED=1` for foreground execution. The original task defaults,
losses, parameter names, optimizer behavior and save intervals are preserved.
See the [ablation guide](src/bridgewam/models/wan22/ablation_bridgewam/README.md)
for the four retained ablation configurations.

## Inference with Your Trained Checkpoints

The `mujoco` environment should ideally stay consistent with the LIBERO data version. Then run LIBERO evaluation:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

The retained `third_party/RoboTwin` files are placeholders in the pinned commit.
Restore the matching working simulator and follow the official instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).
Finish installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/bridgewam_policy" "$(pwd)/third_party/RoboTwin/policy/bridgewam_policy"
```

Then run RoboTwin evaluation:

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```

Common `task_name` examples:

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## Checkpoint Compatibility

The compatibility implementation is copied unchanged from the pinned commit:

- `fastwam` imports, historical class names, Hydra model targets and the old
  `aclation_bridgewam` package spelling resolve through the retained aliases.
  `src/fastwam` is a compatibility entry point, not a duplicate model implementation.
- MoT/BoE parameter registration, checkpoint payloads, key normalization and
  architecture validation are unchanged. Legacy names in checkpoint filenames do
  not require renaming. A checkpoint must still match the selected model topology.
- Registered names such as `mot`, `dit`, `mixtures.video`, `mixtures.action` and
  `latent_bridge_queries` are preserved for weights and optimizer compatibility.
- `model=fastwam_baseline` selects the explicit historical topology: no LBQs,
  full-depth ActionDiT and the direct Video K/V path. It requires a matching
  baseline checkpoint; it does not convert two-layer LBQ weights into a baseline.
- `resume=/path/to/step.pt` loads model weights through the existing trainer.
  An Accelerate/DeepSpeed state directory is the separate full-state resume route;
  its original topology, optimizer and distributed-state constraints still apply.
- The supported `BRIDGEWAM_*` environment variables retain their `FASTWAM_*`
  fallbacks. Upstream download identifiers, licenses and historical resource paths
  retain their original names.

For example, select the explicit fallback with a compatible baseline checkpoint:

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_baseline \
  resume=/path/to/baseline_step.pt
```

This export changes no checkpoint-loading logic and performs no checkpoint
conversion. Full checkpoint and simulator validation should use your verified
runtime and the same architecture configuration as training.

## Acknowledgements

The RoboTwin evaluation code in this repository is adapted from the official [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin). We thank the RoboTwin team for releasing their codebase and assets.

## BibTeX

Upstream FastWAM citation (preserved attribution):

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
