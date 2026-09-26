# BridgeWAM LIBERO-Plus evaluation

This pipeline keeps the original `experiments/libero` inference behavior and uses the
official LIBERO-Plus task classification to evaluate robustness by perturbation and
difficulty. LIBERO-Plus uses one trial per task.

For efficiency, selected tasks are grouped within each suite into balanced worker
shards. `MULTIRUN.tasks_per_worker=50` means one worker loads BridgeWAM once and then
evaluates 49-50 different LIBERO-Plus task variants sequentially. The full benchmark
therefore uses 203 model-loading processes instead of 10,030. Rollout MP4 generation
is disabled by default for Plus and can be restored with
`EVALUATION.save_rollout_video=true`.

## Prerequisites

1. Reuse the verified external LIBERO-Plus checkout and its dependencies. The
   environment is not bundled in this source export. Its assets must be under
   `/path/to/LIBERO-plus/libero/libero/assets`.
2. Install the extra dependencies from that checkout's `extra_requirements.txt`
   in the BridgeWAM environment. Sensor-noise tasks also require ImageMagick.
3. Run from the BridgeWAM export root and pass
   `LIBERO_PLUS.repo_path=/path/to/LIBERO-plus` to the manager or stage launcher.
   The manager creates a per-run LIBERO path configuration and forwards the selected
   environment to workers, leaving `~/.libero/config.yaml` unchanged.

## Stage one

First generate and inspect the complete 10,030-task manifest without launching:

```bash
python experiments/libero-plus/run_libero_plus_manager.py \
  LIBERO_PLUS.repo_path=/path/to/LIBERO-plus \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.create_only=true
```

Run the full official benchmark:

```bash
bash experiments/libero-plus/run_stage1.sh \
  LIBERO_PLUS.repo_path=/path/to/LIBERO-plus \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=6 \
  MULTIRUN.max_tasks_per_gpu=1
```

`run_stage1.sh` requires the full 10,030-task selection by default. For a deterministic
141-task diagnostic pilot, explicitly set `ALLOW_PARTIAL_STAGE1=true` and add
`LIBERO_PLUS.max_tasks_per_cell=1`. This selects at most one task from every suite,
perturbation, and difficulty cell. Filters can use:

```text
LIBERO_PLUS.categories=[camera_viewpoints,robot_initial_states]
LIBERO_PLUS.difficulty_levels=[1,2,3]
LIBERO_PLUS.include_unknown_difficulty=false
```

The main outputs are `task_manifest.jsonl`, `libero_plus_task_results.csv`,
`libero_plus_summary.csv`, and `libero_plus_summary.json`. Missing tasks are reported
separately and are never counted as failed episodes.

`run_stage1.sh` launches the stage-one manager in a detached tmux session by
default, so the scheduler and GPU workers survive an SSH or browser-IDE
disconnect. The launcher prints the tmux session and manager-log paths and also
writes them to `<output_dir>/stage1_launcher_info.txt`. Use `tmux attach -t
<session>` to watch the manager or `tail -f <output_dir>/stage1_manager.log` to
follow its persistent log. Set `LIBERO_PLUS_DETACH=false` for foreground
debugging.
