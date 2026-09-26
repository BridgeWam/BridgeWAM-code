"""Compare actual Git-reference checkpoints with the current implementation.

Run with a CPU PyTorch environment. No pretrained downloads, repository writes,
or trained-checkpoint modifications; all fixtures live in a temporary directory.
This checks tiny models, not the user's full-sized trained checkpoints.
"""
import argparse
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


WORKER = r'''
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.fastwam_idm import FastWAMIDM
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.mot import MoT

torch.set_num_threads(1)
mode, directory, check = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
for kind, cls in [('baseline', FastWAM), ('lbq', FastWAM),
                  ('joint', FastWAMJoint), ('idm', FastWAMIDM)]:
    torch.manual_seed(1207)
    video = WanVideoDiT(hidden_dim=8, in_dim=2, ffn_dim=16, out_dim=2,
        text_dim=8, freq_dim=8, eps=1e-6, patch_size=(1,1,1), num_heads=2,
        attn_head_dim=4, num_layers=2, has_image_input=False,
        seperated_timestep=True, video_attention_mask_mode='first_frame_causal')
    bridge = kind == 'lbq'
    action = ActionDiT(action_dim=2, hidden_dim=8, ffn_dim=16, text_dim=8,
        freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=4, num_layers=2,
        lbq_dim=8 if bridge else None,
        architecture='alternating_cross_self' if bridge else 'full',
        conditioning_mode='lbq_only' if bridge else 'text_state',
        add_pos_embed=bridge, max_action_horizon=16)
    cfg = dict(enabled=bridge, num_lbqs=3, start_layer=0, readout_layer=-1,
        generation_coupling='future_video_reads_lbq', injection_mode='lbq_only',
        lbq_attention='bidirectional', lbq_rope_mode='identity',
        preserve_direct_video_kv=False, freeze_video_expert=False,
        strict_training_scope=False)
    backbone = MoT
    if mode == 'verify' and bridge:
        from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
        backbone = BridgeOfExperts
    mot = backbone({'video':video, 'action':action}, False, latent_bridge_queries=cfg)
    model = cls(video, action, mot, nn.Identity(), text_dim=8, proprio_dim=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = directory / f'old_fastwam_boe_{kind}.pt'
    if mode == 'generate':
        sum(p.square().sum() for p in model.parameters()).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        model.save_checkpoint(path, optimizer=opt, step=5)
    else:
        assert model.load_checkpoint(path, optimizer=opt)['step'] == 5
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    result_path = directory / f'{kind}_expected.pt'
    if mode == 'verify':
        expected = torch.load(result_path, weights_only=True)
        assert expected['initial'].keys() == initial.keys()
        for key, value in initial.items():
            torch.testing.assert_close(expected['initial'][key], value, rtol=0, atol=0)
        if check == 'weights':
            print(f'PASS {kind}: all initial parameters and buffers restored exactly')
            continue
    torch.manual_seed(219)
    latents = torch.randn(2,2,2,2,2)
    data = dict(input_latents=latents, first_frame_latents=latents[:,:,:1],
        context=model.proprio_encoder(torch.randn(2,3,2)),
        context_mask=torch.ones(2,3,dtype=torch.bool), action=torch.randn(2,4,2),
        action_is_pad=None, action_dim_is_pad=None, image_is_pad=None,
        fuse_vae_embedding_in_latents=True)
    with patch.object(model, 'build_inputs', return_value=data):
        loss, metrics = model.training_loss({})
    loss.backward()
    result = dict(initial=initial, loss=loss.detach(),
        gradients={n:None if p.grad is None else p.grad.clone()
                   for n,p in model.named_parameters()},
        parameter_names=list(dict(model.named_parameters())))
    opt.step()
    result['after_step'] = model.state_dict()
    if mode == 'generate':
        torch.save(result, result_path)
    else:
        assert expected['parameter_names'] == result['parameter_names']
        torch.testing.assert_close(expected['loss'], result['loss'], rtol=0, atol=0)
        for section in ['gradients', 'after_step']:
            assert expected[section].keys() == result[section].keys()
            for key, value in expected[section].items():
                torch.testing.assert_close(value, result[section][key], rtol=0, atol=0,
                    msg=lambda message: f'{kind}/{section}/{key}: {message}')
        print(f'PASS {kind}: loss, gradients, parameter order, optimizer continuation (exact CPU equality)')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='1fc36bff9e78a176ad5d0cfdd5e73ef1af29f97c')
    parser.add_argument('--check', choices=['training', 'weights'], default='training',
                        help='Use weights for a reference branch with different training semantics.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.reference + '^{commit}'], cwd=root, text=True,
    ).strip()
    archive = subprocess.check_output(['git', 'archive', commit, 'src'], cwd=root)
    with tempfile.TemporaryDirectory(prefix='bridgewam-history-') as tmp:
        tmp = Path(tmp)
        old = tmp / 'reference'
        old.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(old, filter='data')
        worker = tmp / 'worker.py'
        worker.write_text(WORKER)
        for mode, source in [('generate', old), ('verify', root)]:
            env = dict(os.environ, PYTHONPATH=str(source / 'src'), PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, str(worker), mode, str(tmp), args.check],
                           cwd=tmp, env=env, check=True)
    print(f'Validated reference {commit}')


if __name__ == '__main__':
    main()
