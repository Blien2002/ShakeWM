"""GT reconstruction and fixed-origin C10/H10 four-column WM interpretation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import default_collate
from .data import FrameDataset
from .models import make_decoder, normalize_tokens
from .train import reconstruct, semantic_config
from .metrics import ErrorTotals
from .util import atomic_json


def write_video(path, columns, hz):
    """GIF is portable; lossless PNGs preserve the displayed four columns too."""
    folder = Path(path); folder.mkdir(parents=True, exist_ok=False)
    names = ['C1 true RGB', 'C2 GT latent reconstruction', 'C3 WM prediction reconstruction', 'C4 copy-last reconstruction']
    frames = []
    for i in range(len(columns[0])):
        panels = []
        for name, values in zip(names, columns):
            rgb = (values[i].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            panel = Image.new('RGB', (256, 280), 'white'); panel.paste(Image.fromarray(rgb), (0, 24))
            ImageDraw.Draw(panel).text((4, 6), name, fill='black'); panels.append(panel)
        frame = Image.new('RGB', (1024, 280), 'white')
        for j, panel in enumerate(panels): frame.paste(panel, (256 * j, 0))
        frame.save(folder / f'{i:03d}.png'); frames.append(frame)
    if frames:
        frames[0].save(folder / 'four_columns.gif', save_all=True, append_images=frames[1:],
            duration=round(1000 / hz), loop=0)


@torch.no_grad()
def prediction_report(decoder, frames, cfg, checkpoint_path, wm_config_path, device, output):
    # GT reconstruction is standalone. Native-v2 WM interpretation is optional.
    try:
        from shakewm.checkpointing import load_checkpoint
        from shakewm.validation import validation_inputs
    except ImportError as error:
        raise RuntimeError('WM prediction evaluation requires the native-v2 checkpoint '
                           'and validation interfaces; standalone GT reconstruction is available') from error
    from shakewm.config import Config
    from shakewm.data import WindowDataset
    from shakewm.model import ShakeWM
    from shakewm.engine import precision, to_device
    wm_cfg = Config.load(wm_config_path)
    if (wm_cfg.data.context, wm_cfg.data.horizon, wm_cfg.data.visual_hz) != (10, 10, 10):
        raise ValueError('fixed C10/H10,10Hz evaluation required')
    inputs = cfg['inputs']; binding, norm = validation_inputs(Path(inputs['val_manifest']).parent, wm_cfg)
    dataset = WindowDataset(inputs['val_manifest'], 'val', wm_cfg.data, norm, cache_dir=inputs['val_cache'])
    wm = ShakeWM(wm_cfg.model).to(device).eval().requires_grad_(False)
    state = load_checkpoint(checkpoint_path, wm, wm_cfg, binding['training_manifest_hash'], binding['teacher'],
        device=device, normalization=binding['training_normalization'])
    if state.get('complete_update') is not True or state.get('format') != 'shakewm.checkpoint.v2':
        raise ValueError('complete native WM checkpoint required; no failure/trajectory capsule')
    if dataset.teacher != frames.index['teacher']: raise ValueError('decoder/WM teacher mismatch')
    row_by_id = {row['id']: i for i, row in enumerate(frames.index['episodes'])}
    pairs = ['GT_reconstruction_vs_RGB', 'prediction_vs_RGB', 'prediction_vs_GT_reconstruction',
        'persistence_vs_RGB', 'persistence_vs_GT_reconstruction']
    totals = {key: [ErrorTotals(cfg['roi']) for _ in range(10)] for key in pairs}
    deltas = {key: ErrorTotals(cfg['roi']) for key in pairs}
    latent_sums = [0.0] * 10; counts = [0] * 10; videos = windows = 0
    stride = cfg['evaluation']['origin_stride']; chosen = []; episode_origins = defaultdict(int)
    video_episodes = set(); evaluated_episodes = set()
    for index, (episode, _) in enumerate(dataset.windows):
        if episode_origins[episode] % stride == 0: chosen.append(index)
        episode_origins[episode] += 1
    for index in chosen:
        sample = dataset[index]; batch = to_device(default_collate([sample]), device)
        # Native rollout receives only historical visual/IMU inputs, never targets.
        with precision(device, wm_cfg.train.effective_precision):
            predicted = wm.rollout(batch['history'], batch['short'], batch['long'], batch['eligible'],
                10, dropped=None, normalize_reps=wm_cfg.train.normalize_reps)[0].float()
        if not wm_cfg.train.normalize_reps: raise ValueError('normalized native WM space required')
        episode, origin = dataset.windows[index]
        row = dataset.episodes[episode][0]; source = frames.index['episodes'][row_by_id[row['id']]]
        prepared_episode = row_by_id[row['id']]
        _, rgb = frames.arrays(prepared_episode)
        mask = batch['target_mask'][0].bool()
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not valid_indices: continue
        if any(origin + 1 + k >= len(source['valid']) or not source['valid'][origin + 1 + k] for k in valid_indices):
            raise ValueError('WM mask/cache-to-RGB alignment mismatch')
        if abs(source['times'][origin] - sample['origin_time']) > 1e-8: raise ValueError('origin timestamp mismatch')
        target = normalize_tokens(batch['targets'][0])
        last = normalize_tokens(batch['history'][0, -1:])
        # Predicted features are already normalized by native rollout; do not LN twice.
        c2 = decoder(target[valid_indices]); c3 = decoder(predicted[valid_indices])
        c4 = decoder(last).expand_as(c2)
        c1 = torch.from_numpy(np.stack([np.array(rgb[origin + 1 + k]) for k in valid_indices])).permute(0, 3, 1, 2).to(device).float() / 255
        values = {'GT_reconstruction_vs_RGB': (c2, c1), 'prediction_vs_RGB': (c3, c1),
            'prediction_vs_GT_reconstruction': (c3, c2), 'persistence_vs_RGB': (c4, c1),
            'persistence_vs_GT_reconstruction': (c4, c2)}
        for j, k in enumerate(valid_indices):
            counts[k] += 1; latent_sums[k] += (predicted[k] - target[k]).abs().mean().item()
            for key, (a, b) in values.items(): totals[key][k].add(a[j:j + 1], b[j:j + 1])
            if j and valid_indices[j - 1] + 1 == k:
                for key, (a, b) in values.items(): deltas[key].add((a[j] - a[j - 1])[None], (b[j] - b[j - 1])[None])
        if videos < cfg['evaluation']['video_windows'] and row['id'] not in video_episodes:
            path = output / f'video_{videos:02d}'
            write_video(path, [c1, c2, c3, c4], 10)
            atomic_json(path / 'alignment.json', {'episode_id': row['id'], 'origin_cache_frame': origin,
                'origin_rgb_index': source['rgb_indices'][origin], 'origin_time': sample['origin_time'],
                'horizons': [k + 1 for k in valid_indices],
                'rgb_indices': [source['rgb_indices'][origin + 1 + k] for k in valid_indices],
                'timestamps': [source['times'][origin + 1 + k] for k in valid_indices],
                'display_clamped': True, 'metrics_clamped': False, 'GIF_palette_is_lossy': True})
            videos += 1
            video_episodes.add(row['id'])
        windows += 1
        evaluated_episodes.add(row['id'])
    return {'protocol': 'C10_H10_fixed_origin_free_rollout_only_past_inputs', 'wm_checkpoint': str(checkpoint_path),
        'wm_global_step': state['global_step'], 'wm_parameters_updated': False, 'decoder_updated': False,
        'eligible_windows': len(dataset), 'evaluated_windows': windows, 'origin_stride': stride,
        'eligible_episodes': len(episode_origins), 'evaluated_episode_ids': sorted(evaluated_episodes),
        'scope': 'all_val_episodes_subsampled_origins', 'valid_counts_by_horizon': counts,
        'latent_l1_by_horizon': [value / count if count else None for value, count in zip(latent_sums, counts)],
        'RGB_pairs_by_horizon': {key: [v.report() for v in items] for key, items in totals.items()},
        'within_rollout_adjacent_valid_frame_delta': {key: value.report() for key, value in deltas.items()},
        'roi_kind': 'predefined_rectangle_not_object_mask', 'roi_xyxy_normalized': cfg['roi']}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--wm-checkpoint'); p.add_argument('--wm-config'); p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    cfg = json.loads(Path(a.config).read_text()); device = torch.device(a.device)
    output = Path(a.output); output.mkdir(parents=True, exist_ok=False)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if state.get('format') != 'shakewm.decoder.complete.v1' or not state.get('complete_update'):
        raise ValueError('complete decoder checkpoint required')
    if state['config'] != semantic_config(cfg): raise ValueError('decoder configuration differs')
    frames = FrameDataset(Path(cfg['inputs']['val_frames']) / 'index.json')
    if frames.index['split'] != 'val' or len(frames.index['episodes']) != 45:
        raise ValueError('complete fixed real val45 required')
    if state['teacher'] != frames.index['teacher'] or state['val_identity'] != frames.index['identity']:
        raise ValueError('decoder validation split/teacher differs')
    decoder = make_decoder(cfg).to(device).eval().requires_grad_(False)
    decoder.load_state_dict(state['model'], strict=True)
    report = reconstruct(decoder, frames, device, cfg['train']['eval_batch_size'], cfg['roi'])
    atomic_json(output / 'GT_reconstruction.json', report)
    if bool(a.wm_checkpoint) != bool(a.wm_config): raise ValueError('WM checkpoint and frozen config must be supplied together')
    if a.wm_checkpoint:
        report = prediction_report(decoder, frames, cfg, a.wm_checkpoint, a.wm_config, device, output)
        atomic_json(output / 'four_columns.json', report)
    atomic_json(output / 'receipt.json', {'phase': 'completed', 'GT_reconstruction': True,
        'WM_prediction_evaluated': bool(a.wm_checkpoint), 'parameters_updated': False})


if __name__ == '__main__': main()
