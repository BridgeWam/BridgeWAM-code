"""Explicit Bridge-of-Experts (BoE) interface, sharing the proven MoT kernels.

BoE names a topology: Video -> LBQ -> Action, without direct Video K/V.
It adds no parameters or buffers; the serialized ``mot.mixtures.*`` contract
and parameter registration order remain identical to the historical model.
"""
from .mot import MoT


class BridgeOfExperts(MoT):
    def __init__(self, mixtures, mot_checkpoint_mixed_attn=True,
                 action_video_kv_routing=None, latent_bridge_queries=None):
        cfg = latent_bridge_queries or {}
        if not cfg.get('enabled', False) or cfg.get('preserve_direct_video_kv', True):
            raise ValueError('BridgeOfExperts requires enabled LBQs and preserve_direct_video_kv=false.')
        super().__init__(mixtures, mot_checkpoint_mixed_attn,
                         action_video_kv_routing, latent_bridge_queries)

    def forward_bridge_video(self, **kwargs):
        """Train the video/LBQ stream without collecting direct Video K/V."""
        kwargs['collect_video_kv_cache'] = False
        return self.forward_video_with_lbqs(**kwargs)

    def prefill_bridge(self, **kwargs):
        """Compute reusable LBQ conditioning for action denoising."""
        kwargs['collect_video_kv_cache'] = False
        return self.prefill_video_cache_with_lbqs(**kwargs)

    def forward_bridge_action(self, **kwargs):
        """Denoise actions from projected LBQs using the original math."""
        if kwargs.get('video_kv_cache'):
            raise ValueError('BridgeOfExperts does not consume direct Video K/V.')
        return self.forward_action_with_video_cache(**kwargs)


# Short architectural name, not a second implementation.
BoE = BridgeOfExperts
