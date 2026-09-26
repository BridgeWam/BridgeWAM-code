# BridgeWAM on LIBERO-Pro

This evaluator uses the pre-generated task interface in the
[official LIBERO-Pro code](https://github.com/Zxy-MLlab/LIBERO-PRO).
It reuses model loading, image/state processing, action denormalization, gripper
conversion, action chunks, rollouts and video writing from
`experiments/libero/eval_libero_single.py`. Simulator code and resources come from
an external LIBERO-PRO checkout; they are not bundled in this source export.

## Evaluation scope and protocol

Supported base suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

| `LIBERO_PRO.perturbations` | Suite suffix | Perturbation |
|---|---|---|
| `object` | `_object` | Objects |
| `position` | `_swap` | Positions |
| `language` | `_lan` | Language |
| `task` | `_task` | Tasks |
| `environment` | `_env` | Environment |

Task IDs and ordering are read from the selected environment's
`benchmark/libero_suite_task_map.py` and recorded in the manifest. The task table
used by the pinned version has ten tasks per combination: 200 tasks total, with
50 initial states per task by default (10,000 episodes). Each task must provide
enough pre-generated initial states; missing states cause failure rather than
repeated use of existing states. These are five separate perturbation categories.
The evaluator does not generate combined perturbations at runtime or modify the
external `evaluation_config.yaml`.

Instructions come from the environment's parsed BDDL `:language`, preserving
language/task perturbations instead of deriving text from filenames. Default
episode horizons follow the reference protocol: Spatial 220, Object 280, Goal 300,
and LIBERO-10 520. There are ten additional wait steps; the BridgeWAM replanning
interval remains ten steps. Override `EVALUATION.max_steps`,
`EVALUATION.num_steps_wait` or `EVALUATION.replan_steps` explicitly if needed.
These settings are recorded in the run configuration and should stay fixed for
comparable results.

## Environment and resources

Use a Python environment with BridgeWAM and the matching LIBERO/robosuite/MuJoCo
dependencies. Obtain complete `bddl_files` and `init_files` from the
[official dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro), and
prepare the simulator assets:

```text
/path/to/LIBERO-PRO/
`-- libero/libero/
    |-- benchmark/libero_suite_task_map.py
    |-- bddl_files/libero_goal_lan/<task>.bddl
    |-- init_files/libero_goal_lan/<task>.pruned_init
    `-- assets/
```

The other 19 suite directories follow the same layout. Before launching workers,
the manager checks that each BDDL/init file exists and is nonempty. Every run gets
its own `libero_pro_config/config.yaml`. The worker checks that the imported LIBERO
module and resource paths belong to the selected Pro checkout, preventing accidental
use of standard LIBERO, Plus or the user's default `~/.libero` configuration.

## Generate a task plan only

Run from the BridgeWAM export root. This requires no checkpoint, GPU or MuJoCo.
It validates the task table and selection, not simulator resource completeness.

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_PRO_REPO=/path/to/LIBERO-PRO

python experiments/libero-pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true \
  LIBERO_PRO.expected_num_tasks=200 \
  EVALUATION.output_dir=/path/to/pro-plan
```

## Eight-GPU evaluation

Select a task whose architecture matches the checkpoint. The default is the main
32-LBQ experiment with 30 Video layers and two Action layers. Model code comes
from this BridgeWAM checkout; `LIBERO_PRO_REPO` selects only the simulator environment.
Alternatively, pass `LIBERO_PRO.repo_path=/path/to/LIBERO-PRO`.

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_PRO_REPO=/path/to/LIBERO-PRO
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/libero-pro/run_libero_pro_manager.py \
  task=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4 \
  ckpt=/path/to/step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.output_dir=/path/to/new-pro-run \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1 \
  LIBERO_PRO.expected_num_tasks=200
```

The manager runs in the foreground and waits for all workers. Use tmux yourself
when background execution is needed. It uses the current Python by default and
also accepts `BRIDGEWAM_PYTHON` or legacy `FASTWAM_PYTHON`. GPUs are selected from
`CUDA_VISIBLE_DEVICES`; each worker sees only its assigned GPU. Allocate idle GPUs:
the manager does not manage other processes' GPU usage. Each worker loads a model
and evaluates one task.

To evaluate a subset of tasks and perturbations:

```bash
python experiments/libero-pro/run_libero_pro_manager.py \
  ckpt=/path/to/step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.output_dir=/path/to/pro-language-smoke \
  'MULTIRUN.task_suite_names=[libero_goal]' \
  'MULTIRUN.task_ids=[0]' \
  'LIBERO_PRO.perturbations=[language]' \
  MULTIRUN.num_gpus=1
```

A worker can also be launched directly:

```bash
python experiments/libero-pro/eval_libero_pro_single.py \
  ckpt=/path/to/step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.output_dir=/path/to/pro-single \
  EVALUATION.task_suite_name=libero_goal_lan \
  EVALUATION.task_id=0
```

## Outputs and completeness

- `worker_config.yaml`: the complete composed model/evaluation configuration loaded by workers.
- `task_manifest.jsonl`, `task_manifest.sha256`, `tasks.txt`: ordered tasks and manifest hash.
- `libero_pro_config/config.yaml`: environment paths isolated to this run.
- `task_logs/`: per-task logs.
- `<suite>/gpu*_task*_results.json`: results, actual language, Pro identity and episode horizon.
- `manager_runtime.json`: start/end times, base Git commit when available, checkpoint path and worker exit codes.
- `summary.json`: overall, base-suite, perturbation and 20 combination summaries.

When running from a source archive without Git metadata, the runtime's Git commit
field may be null. Do not include identifying local paths or Git metadata when
sharing run artifacts for anonymous review.

Success rates aggregate total successful episodes divided by total evaluated
episodes. Missing tasks mark a run incomplete; duplicate results or mismatched task
identities fail explicitly. The manager exits zero only when every worker succeeds
and all results are present. Use a fresh output directory for each experiment;
do not combine results from different checkpoints.

```bash
python experiments/libero-pro/summarize_results.py --output_dir=/path/to/pro-run
```

The full repository's tests cover manifests, environment selection, BDDL language,
the shared evaluator interface, GPU arguments, configuration transfer and summary
validation. Those tests are outside this source export. CPU/mock checks do not
establish real-checkpoint success rates in MuJoCo or on GPUs.
