# BridgeWAM

BridgeWAM connects a pretrained video diffusion backbone to an action diffusion
head through **Latent Bridge Queries (LBQs)**. This repository contains model
implementations, training configurations, checkpoint compatibility utilities, and
simulation evaluation entry points for LIBERO, LIBERO-Plus, LIBERO-Pro, and RoboTwin.

## Scope of this release

This is a source release. Datasets, pretrained weights, trained checkpoints,
simulation assets, experiment logs, and the full regression test suite are not
included. LIBERO environments must be installed separately.

**The bundled `third_party/RoboTwin` tree contains empty placeholder files, not a
working simulator.** Restore a complete compatible RoboTwin installation and its
assets before using the RoboTwin evaluator. The presence of an evaluation script
does not establish that the corresponding environment is ready to run.

The primary implementation lives in `src/bridgewam`. The `fastwam` namespace and
selected legacy model names remain available for checkpoint and configuration
compatibility. They do not select a different implementation by themselves.

## Repository layout

```text
configs/                         Model, dataset, training, and evaluation settings
experiments/libero/               Standard LIBERO evaluation
experiments/libero-plus/          LIBERO-Plus task manifests and sharded evaluation
experiments/libero-pro/           LIBERO-Pro perturbation evaluation
experiments/robotwin/             RoboTwin evaluation and policy adapters
scripts/                         Training, preprocessing, and profiling utilities
src/bridgewam/                   Models, datasets, runtime, and trainer
src/fastwam/                     Legacy import compatibility
src/latent_bridge_queries/       LBQ components
third_party/RoboTwin/            Simulator placeholders; external setup required
```

## Installation

Training and simulation evaluation target Linux with NVIDIA GPUs. Install a
compatible NVIDIA driver, Conda, `tmux`, and the `flock` utility. The dependency
pins in `pyproject.toml` target Python 3.10 or later and PyTorch with CUDA 12.8.

```bash
conda create -n bridgewam python=3.10 -y
conda activate bridgewam
python -m pip install --upgrade pip
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

Run the following setup and subsequent commands from the repository root:

```bash
export CODE_ROOT="$PWD"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BRIDGEWAM_PYTHON="$(command -v python)"
export DIFFSYNTH_MODEL_BASE_PATH="$CODE_ROOT/checkpoints"
export BRIDGEWAM_TRAIN_OUTPUT_BASE="$CODE_ROOT/runs/train"
export WANDB_MODE=disabled
```

W&B logging is disabled in the default training configuration. Enable it only
after configuring your own account; do not publish logs or account metadata as
part of an anonymous source submission.

## Pretrained models and data

### Model components

Prepare the Wan2.2-TI2V-5B backbone, compatible VAE, text encoder, and tokenizer.
With the default `model.redirect_common_files=true`, the loader expects the
following structure under `DIFFSYNTH_MODEL_BASE_PATH`:

```text
checkpoints/
|-- Wan-AI/Wan2.2-TI2V-5B/
|   `-- diffusion_pytorch_model*.safetensors
|-- Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
`-- DiffSynth-Studio/Wan-Series-Converted-Safetensors/
    |-- Wan2.2_VAE.safetensors
    `-- models_t5_umt5-xxl-enc-bf16.safetensors
```

The loader supports downloading missing upstream components. For an offline run,
prepare every required component first, then set:

```bash
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

Create the interpolated ActionDiT backbone with the base model configuration,
which retains the full layer stack needed by the preprocessing utility:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/bridgewam.yaml \
  --output "$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  --device cuda \
  --dtype bfloat16

export ACTION_DIT_PRETRAINED_PATH="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
```

### Dataset layout

Use datasets compatible with the included LeRobot-based loader and processing
configurations. The default paths are relative to the repository root:

```text
data/
|-- libero_mujoco3.3.2/
|   |-- libero_spatial_no_noops_lerobot/
|   |-- libero_object_no_noops_lerobot/
|   |-- libero_goal_no_noops_lerobot/
|   `-- libero_10_no_noops_lerobot/
|-- robotwin2.0/
|   |-- robotwin2.0/              # Dataset containing data/, meta/, and videos/
|   `-- dataset_stats.json
`-- text_embeds_cache/
    |-- libero/
    `-- robotwin/
```

Override `data.train.dataset_dirs` and the corresponding validation paths when
using a different location. Keep action/state normalization statistics paired
with the dataset and checkpoint that produced them.

The main training task uses cached text embeddings instead of loading the text
encoder in each training process. Generate the cache after installing the data
and model components:

```bash
TASK=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4
python scripts/precompute_text_embeds.py task="$TASK"
```

## Main LIBERO training configuration

The main task above uses the following settings:

| Component | Configuration |
|---|---|
| Video backbone | Wan2.2-TI2V-5B, 30 layers, hidden dimension 3072 |
| Action head | Two alternating cross-attention/self-attention layers, hidden dimension 1024 |
| Bridge queries | 32 LBQs, introduced at layer 0, read out at the final video layer |
| LBQ routing | `lbq_only`, bidirectional attention, identity RoPE |
| Video coupling | `future_video_reads_lbq` |
| Optimizer schedule | Learning rate `1e-4`, cosine schedule, 10 epochs |
| Batch size | 16 per process, gradient accumulation 1 |

At training initialization, the Video DiT loads pretrained Wan weights and the
ActionDiT backbone loads the preprocessed weights. LBQs and task-specific
adapters are newly initialized. The VAE is frozen. The main task fine-tunes the
video and action experts; frozen-video, joint, and IDM variants are separate tasks.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_ID="bridgewam_libero_$(date +%Y%m%d_%H%M%S)"
export BRIDGEWAM_TMUX_SESSION_NAME="$RUN_ID"

bash scripts/train_zero1.sh 8 \
  task="$TASK" \
  seed=42 \
  num_workers=8 \
  save_every=5000 \
  model.skip_dit_load_from_pretrain=false \
  model.latent_bridge_queries.num_lbqs=32 \
  model.latent_bridge_queries.start_layer=0 \
  model.latent_bridge_queries.readout_layer=-1
```

The launcher creates a detached tmux session and prints the log location. Unless
`output_dir` is explicitly overridden, artifacts are written to:

```text
runs/train/<task>/<run_id>/
|-- train.log
|-- dataset_stats.json
`-- checkpoints/weights/step_<number>.pt
```

Set `BRIDGEWAM_TMUX_DISABLED=1` for foreground training. Legacy `FASTWAM_*`
launcher variables remain supported, with `BRIDGEWAM_*` taking precedence.

## Evaluation

Use the same model architecture as the checkpoint's training configuration.
Prepare a checkpoint and its matching statistics, then set:

```bash
export CKPT=/path/to/checkpoint.pt
export STATS=/path/to/dataset_stats.json
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES=all
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TASK=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4
```

The examples below start with one worker per GPU. Increase
`MULTIRUN.max_tasks_per_gpu` only after confirming sufficient GPU and host memory.
Each worker loads its own model. Allocate resources before starting another
benchmark; these managers are not a general cross-benchmark GPU scheduler.

### LIBERO

Install a compatible [LIBERO environment](https://github.com/Lifelong-Robot-Learning/LIBERO)
and configure its BDDL files, initial states, and assets. Point the import path and
`LIBERO_CONFIG_PATH` at that installation, not at a Plus or Pro installation.

```bash
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_CONFIG_PATH=/path/to/libero-config
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT:$LIBERO_ROOT"

python experiments/libero/run_libero_manager.py \
  task="$TASK" \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.output_dir="$CODE_ROOT/runs/eval/libero/$(date +%Y%m%d_%H%M%S)" \
  EVALUATION.num_trials=50 \
  model.skip_dit_load_from_pretrain=true \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

The default selection contains four suites with ten tasks each. The manager
waits for its workers; run the manager inside tmux if it must survive a disconnect.

### LIBERO-Plus

Install the separate LIBERO-Plus environment and its additional dependencies.
The stage launcher requires the full 10,030-task selection by default and starts
a detached manager. Each task uses one trial.

```bash
bash experiments/libero-plus/run_stage1.sh \
  LIBERO_PLUS.repo_path=/path/to/LIBERO-plus \
  task="$TASK" \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.output_dir="$CODE_ROOT/runs/eval/libero-plus/$(date +%Y%m%d_%H%M%S)" \
  EVALUATION.num_trials=1 \
  model.skip_dit_load_from_pretrain=true \
  MULTIRUN.tasks_per_worker=50 \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

See the [LIBERO-Plus guide](experiments/libero-plus/README.md) for manifests,
diagnostic subsets, environment isolation, and summary files.

### LIBERO-Pro

Prepare an external LIBERO-Pro checkout with all BDDL files, initial states, and
assets. The current evaluator reads resources from that checkout; it does not
generate perturbed environments. Keep the same resource variant across models.

```bash
export LIBERO_PRO_REPO=/path/to/LIBERO-PRO

python experiments/libero-pro/run_libero_pro_manager.py \
  task="$TASK" \
  ckpt="$CKPT" \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.output_dir="$CODE_ROOT/runs/eval/libero-pro/$(date +%Y%m%d_%H%M%S)" \
  EVALUATION.num_trials=50 \
  model.skip_dit_load_from_pretrain=true \
  'LIBERO_PRO.perturbations=[environment,position,object,language,task]' \
  LIBERO_PRO.expected_num_tasks=200 \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

This manager runs in the foreground; use tmux for persistent execution. See the
[LIBERO-Pro guide](experiments/libero-pro/README.md) for task planning, resource
validation, episode horizons, and completeness checks.

### RoboTwin

Restore the complete simulator under `third_party/RoboTwin` before running
`experiments/robotwin/run_robotwin_manager.py`. In particular, the manager reads
`third_party/RoboTwin/task_config/_eval_step_limit.yml` from this checkout even
when `EVALUATION.robotwin_root` is overridden. Follow the upstream
[RoboTwin installation instructions](https://github.com/RoboTwin-Platform/RoboTwin)
for assets and simulation dependencies.

Select a task and overrides matching the RoboTwin checkpoint. Do not use a
LIBERO checkpoint or assume that the baseline RoboTwin task enables the two-layer
LBQ architecture. See `configs/sim_robotwin.yaml` for evaluation settings,
including episode count and instruction type.

## Checkpoint compatibility and result integrity

- Legacy import and state-dictionary names are supported where implemented in
  `src/bridgewam/_legacy_imports.py` and
  `src/bridgewam/models/wan22/checkpoint_compat.py`.
- Compatibility does not make different query counts, layer counts, readout
  layers, action dimensions, or model variants interchangeable.
- `model.skip_dit_load_from_pretrain=true` skips backbone preloading during
  evaluation; the trained checkpoint must then restore the required weights.
  It does not mean that random backbone weights are intended for evaluation.
- Use a fresh output directory for each checkpoint/configuration pair. A summary
  file or a live tmux session alone does not prove a complete evaluation. Check
  worker errors, expected task counts, result coverage, and manager exit status.
- `scripts/verify_bridgewam_migration.py` requires a separate repository with
  suitable Git history and an explicit `--reference`; it is not a standalone
  checkpoint or simulator validation test.

## Anonymous distribution

Distribute only reviewed source files. Git history, remotes, credentials, local
configuration, training logs, checkpoint metadata, and generated outputs may
contain identifying information and must be reviewed separately. `.gitignore`
does not remove files already committed to history. Avoid adding personal
accounts, institution-specific paths, tracking badges, or author-profile links.

## License and third-party components

See [LICENSE](LICENSE). This implementation builds on FastWAM and uses components
from Wan/DiffSynth, LeRobot, and the supported simulation benchmarks. Existing
third-party copyright and license notices are retained; they describe upstream
provenance and are not a declaration of this submission's authorship. External
code, datasets, and model weights remain subject to their respective licenses.
