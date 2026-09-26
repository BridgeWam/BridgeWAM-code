import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.mot import MoT
from bridgewam.models.wan22.wan_video_dit import WanVideoDiT
from bridgewam.runtime import _mixed_precision_to_model_dtype
from bridgewam.utils.config_resolvers import register_default_resolvers


@dataclass
class BenchResult:
    name: str
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_min: float
    latency_ms_max: float
    flops: Optional[int]
    cuda_max_memory_mb: Optional[float]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_callable(
    name: str,
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
    measure_flops: bool,
) -> BenchResult:
    for _ in range(warmup):
        fn()
    _sync(device)

    latencies = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    flops = None
    if measure_flops:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        _sync(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            fn()
            _sync(device)
        flops = int(sum(evt.flops for evt in prof.key_averages() if evt.flops is not None))

    lat = torch.tensor(latencies, dtype=torch.float64)
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return BenchResult(
        name=name,
        latency_ms_mean=float(lat.mean().item()),
        latency_ms_p50=float(lat.quantile(0.50).item()),
        latency_ms_p90=float(lat.quantile(0.90).item()),
        latency_ms_min=float(lat.min().item()),
        latency_ms_max=float(lat.max().item()),
        flops=flops,
        cuda_max_memory_mb=peak_mb,
    )


def _build_mot_attention_mask(
    video_expert: WanVideoDiT,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
    *,
    joint_action_attends_full_video: bool,
) -> torch.Tensor:
    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = video_expert.build_video_to_video_mask(
        video_seq_len=video_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        device=device,
    )
    mask[video_seq_len:, video_seq_len:] = True
    if joint_action_attends_full_video:
        mask[video_seq_len:, :video_seq_len] = True
    else:
        mask[video_seq_len:, : min(video_tokens_per_frame, video_seq_len)] = True
    return mask


def _config_to_dict(cfg_node):
    return OmegaConf.to_container(cfg_node, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile BridgeWAM video/action DiT latency and torch.profiler FLOPs with synthetic tensors."
    )
    parser.add_argument("--config-name", default="train", help="Hydra config name under configs/.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, e.g. task=libero_joint_2cam224_1e-4 or model=bridgewam_idm.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", default=None, choices=[None, "no", "fp16", "bf16"])
    parser.add_argument("--height", type=int, default=None, help="Input image height before VAE downsample.")
    parser.add_argument("--width", type=int, default=None, help="Input image width before VAE downsample.")
    parser.add_argument("--num-video-frames", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--flops", action="store_true", help="Run one torch.profiler pass with with_flops=True.")
    parser.add_argument(
        "--joint-action-attends-full-video",
        action="store_true",
        help="Match BridgeWAMJoint action->video mask. Default matches BridgeWAM first-frame-only mask.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.override)
    OmegaConf.resolve(cfg)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    mixed_precision = args.mixed_precision or str(cfg.get("mixed_precision", "bf16"))
    dtype = _mixed_precision_to_model_dtype(mixed_precision)

    video_cfg = _config_to_dict(cfg.model.video_dit_config)
    action_cfg = _config_to_dict(cfg.model.action_dit_config)
    video_cfg["use_gradient_checkpointing"] = False
    action_cfg["use_gradient_checkpointing"] = False

    video_expert = WanVideoDiT(**video_cfg).to(device=device, dtype=dtype).eval()
    action_expert = ActionDiT(**action_cfg).to(device=device, dtype=dtype).eval()
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    ).to(device=device, dtype=dtype).eval()

    height = args.height or int(cfg.data.train.video_size[0])
    width = args.width or int(cfg.data.train.video_size[1])
    num_video_frames = args.num_video_frames or int(cfg.data.train.num_frames)
    action_horizon = args.action_horizon or int((num_video_frames - 1) * cfg.data.train.action_video_freq_ratio)
    context_len = args.context_len or int(cfg.data.train.get("context_len", cfg.model.get("tokenizer_max_len", 128)))

    vae_downsample = 16
    temporal_downsample = 4
    latent_t = (num_video_frames - 1) // temporal_downsample + 1
    latent_h = height // vae_downsample
    latent_w = width // vae_downsample
    if height % vae_downsample != 0 or width % vae_downsample != 0:
        raise ValueError(f"height/width must be divisible by {vae_downsample}, got {height}x{width}.")
    if (num_video_frames - 1) % temporal_downsample != 0:
        raise ValueError(
            f"num_video_frames must satisfy (T-1) % {temporal_downsample} == 0, got {num_video_frames}."
        )

    bsz = int(args.batch_size)
    latents_video = torch.randn(
        (bsz, int(video_cfg["in_dim"]), latent_t, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )
    first_frame_latents = latents_video[:, :, 0:1].contiguous()
    latents_action = torch.randn(
        (bsz, action_horizon, int(action_cfg["action_dim"])),
        device=device,
        dtype=dtype,
    )
    context = torch.randn((bsz, context_len, int(video_cfg["text_dim"])), device=device, dtype=dtype)
    context_mask = torch.ones((bsz, context_len), device=device, dtype=torch.bool)
    timestep_video = torch.full((bsz,), 0.5, device=device, dtype=dtype)
    timestep_action = torch.full((bsz,), 0.5, device=device, dtype=dtype)
    timestep_zero = torch.zeros((bsz,), device=device, dtype=dtype)

    with torch.no_grad():
        action_conditioning = action_expert.prepare_conditioning(
            text_state_context=context,
            text_state_mask=context_mask,
        )
        video_pre = video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_conditioning["cross_context"],
            context_mask=action_conditioning["cross_mask"],
        )
        attention_mask = _build_mot_attention_mask(
            video_expert,
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=device,
            joint_action_attends_full_video=bool(args.joint_action_attends_full_video),
        )

        first_video_pre = video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_zero,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        first_attention_mask = _build_mot_attention_mask(
            video_expert,
            video_seq_len=first_video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(first_video_pre["meta"]["tokens_per_frame"]),
            device=device,
            joint_action_attends_full_video=False,
        )
        video_kv_cache = mot.prefill_video_cache(
            video_tokens=first_video_pre["tokens"],
            video_freqs=first_video_pre["freqs"],
            video_t_mod=first_video_pre["t_mod"],
            video_context_payload={
                "context": first_video_pre["context"],
                "mask": first_video_pre["context_mask"],
            },
            video_attention_mask=first_attention_mask[: first_video_pre["tokens"].shape[1], : first_video_pre["tokens"].shape[1]],
        )

    @torch.no_grad()
    def video_forward():
        return video_expert(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )

    @torch.no_grad()
    def action_forward():
        return action_expert(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

    @torch.no_grad()
    def mot_joint_step():
        vp = video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        ap = action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_conditioning["cross_context"],
            context_mask=action_conditioning["cross_mask"],
        )
        out = mot(
            embeds_all={"video": vp["tokens"], "action": ap["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": vp["freqs"], "action": ap["freqs"]},
            context_all={
                "video": {"context": vp["context"], "mask": vp["context_mask"]},
                "action": {"context": ap["context"], "mask": ap["context_mask"]},
            },
            t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]},
        )
        return video_expert.post_dit(out["video"], vp), action_expert.post_dit(out["action"], ap)

    @torch.no_grad()
    def action_cached_step():
        ap = action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_conditioning["cross_context"],
            context_mask=action_conditioning["cross_mask"],
        )
        out = mot.forward_action_with_video_cache(
            action_tokens=ap["tokens"],
            action_freqs=ap["freqs"],
            action_t_mod=ap["t_mod"],
            action_context_payload={"context": ap["context"], "mask": ap["context_mask"]},
            video_kv_cache=video_kv_cache,
            attention_mask=first_attention_mask,
            video_seq_len=int(first_video_pre["tokens"].shape[1]),
        )
        return action_expert.post_dit(out, ap)

    bench_items = [
        ("video_expert_full_step", video_forward),
        ("action_expert_standalone_step", action_forward),
        ("mot_joint_video_action_step", mot_joint_step),
        ("mot_action_step_with_first_frame_video_cache", action_cached_step),
    ]
    results = [
        _time_callable(
            name,
            fn,
            device=device,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            measure_flops=bool(args.flops),
        )
        for name, fn in bench_items
    ]

    payload = {
        "shape": {
            "batch_size": bsz,
            "height": height,
            "width": width,
            "num_video_frames": num_video_frames,
            "latent_shape": list(latents_video.shape),
            "video_seq_len": int(video_pre["tokens"].shape[1]),
            "first_frame_video_seq_len": int(first_video_pre["tokens"].shape[1]),
            "video_tokens_per_frame": int(video_pre["meta"]["tokens_per_frame"]),
            "action_horizon": action_horizon,
            "action_seq_len": int(action_pre["tokens"].shape[1]),
            "context_len": context_len,
            "dtype": str(dtype),
            "device": str(device),
        },
        "results": [asdict(r) for r in results],
    }

    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
