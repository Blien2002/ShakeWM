"""Unique valid frames, existing split/cache, and CPU-only RGB materialization."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from shakewm.config import digest_json
from .util import atomic_json


def select_frames(times, valid, end_time, acquisition_hz=20, visual_hz=10):
    if acquisition_hz % visual_hz or len(times) != len(valid):
        raise ValueError('invalid acquisition/visual grid')
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / acquisition_hz, atol=1e-5, rtol=0):
        raise ValueError('RGB acquisition-grid mismatch')
    selected = np.arange(0, len(times), acquisition_hz // visual_hz)
    keep = np.asarray(valid[selected], bool) & (times[selected] <= float(end_time) + 1e-9)
    return selected, keep


def safe_child(root, name):
    p = (Path(root) / name).resolve()
    if not p.is_relative_to(Path(root).resolve()): raise ValueError('episode path escapes data root')
    return p


def input_contract(manifest_path, cache_path, split):
    manifest = json.loads(Path(manifest_path).read_text())
    cache = json.loads((Path(cache_path) / 'index.json').read_text())
    if manifest.get('synthetic') is not False or manifest.get('schema') != 'shakewm.manifest.v1':
        raise ValueError('real normalized manifest required')
    rows = manifest['episodes']
    if not rows or any(r['split'] != split for r in rows):
        raise ValueError('expected a complete isolated split view')
    if cache.get('manifest_hash') != digest_json(manifest) or cache.get('camera') != manifest.get('camera'):
        raise ValueError('existing cache does not bind this manifest/camera')
    teacher = cache['teacher']
    if cache.get('schema') != 'shakewm.cache.v1' or teacher.get('kind') != 'official_vjepa2_1_vit_base_384':
        raise ValueError('official V-JEPA2.1 indexed cache required')
    if cache['teacher_hash'] != digest_json(teacher) or (teacher.get('patches'), teacher.get('feature_dim'),
        teacher.get('token_order'), teacher.get('temporal_mode'), teacher.get('input_size')) != (256, 768, 'row_major', 'single_image_T1', 256):
        raise ValueError('expected exact official single-image 256x768 teacher contract')
    if cache['visual_hz'] != 10 or set(cache['episodes']) != {r['id'] for r in rows}:
        raise ValueError('10Hz complete cache required')
    for row in rows:
        if cache['episodes'][row['id']]['source_sha256'] != row['source_sha256']:
            raise ValueError('source/cache metadata binding mismatch')
    return manifest, cache


def prepare_split(manifest_path, cache_path, output, split, expected_episodes):
    """One CPU pass over existing NPZ. No encoder, migration, WM update or crop."""
    manifest, cache = input_contract(manifest_path, cache_path, split)
    if len(manifest['episodes']) != expected_episodes:
        raise ValueError('complete train360/val45 required')
    output, source = Path(output), Path(manifest_path).resolve().parent
    identity = digest_json({'manifest': manifest, 'cache': cache, 'split': split,
        'rgb_mapping': 'cache_j_to_rgb_2j', 'target': 'RGB_uint8_256_no_crop'})
    if (output / 'index.json').exists():
        previous = json.loads((output / 'index.json').read_text())
        if previous['identity'] != identity: raise ValueError('prepared RGB belongs to different inputs')
        return previous
    output.mkdir(parents=True, exist_ok=True)
    index = {'format': 'shakewm.decoder.frames.v1', 'identity': identity, 'split': split,
        'teacher': cache['teacher'], 'teacher_hash': cache['teacher_hash'], 'camera': cache['camera'],
        'manifest': str(Path(manifest_path).resolve()), 'cache': str(Path(cache_path).resolve()),
        'feature_space': 'FP32_nonaffine_token_LN', 'episodes': [], 'valid_frames': 0}
    for i, row in enumerate(manifest['episodes']):
        rgbpath, metapath = output / f'{i:06d}.rgb.npy', output / f'{i:06d}.meta.json'
        feature_path = safe_child(cache_path, cache['episodes'][row['id']]['path'])
        z = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        with np.load(safe_child(source, row['path']), allow_pickle=False) as raw:
            rgb, times = raw['rgb'], raw['rgb_time']
            if rgb.dtype != np.uint8 or rgb.shape[1:] != (256, 256, 3):
                raise ValueError('RGB must match teacher input without resizing/cropping')
            selected, keep = select_frames(times, raw['rgb_valid'], raw['end_time'])
            if z.shape != (len(selected), 256, 768): raise ValueError('cache frame/geometry mismatch')
            meta = {'id': row['id'], 'split': split, 'state_id': row['state_id'],
                'state_fingerprint': row.get('state_fingerprint'), 'source_sha256': row['source_sha256'],
                'feature_sha256': cache['episodes'][row['id']]['sha256'], 'features': str(feature_path),
                'rgb': str(rgbpath.resolve()), 'rgb_indices': selected.tolist(),
                'times': times[selected].tolist(), 'valid': keep.tolist(), 'shape': [len(selected), 256, 256, 3]}
            if metapath.exists():
                if json.loads(metapath.read_text()) != meta or not rgbpath.exists():
                    raise ValueError('incomplete or mismatched prepared episode; preserve and report')
            else:
                # A partial file from interruption is retained, never consumed as complete.
                part = rgbpath.with_suffix(rgbpath.suffix + '.partial')
                if part.exists() or rgbpath.exists(): raise FileExistsError('unpublished RGB materialization exists')
                array = np.lib.format.open_memmap(part, mode='w+', dtype=np.uint8, shape=tuple(meta['shape']))
                array[:] = rgb[selected]; array.flush(); del array
                part.rename(rgbpath)
                atomic_json(metapath, meta)
        index['episodes'].append(meta); index['valid_frames'] += int(sum(meta['valid']))
        print(json.dumps({'prepared_split': split, 'episode': i + 1, 'valid_frames': sum(meta['valid'])}), flush=True)
    if not index['valid_frames']: raise ValueError('no valid real frames')
    atomic_json(output / 'index.json', index)
    return index


def check_split_isolation(train, val):
    if train['teacher'] != val['teacher'] or train['camera'] != val['camera']:
        raise ValueError('train/val teacher or camera mismatch')
    for key in ['id', 'state_id', 'state_fingerprint']:
        a = {str(r[key]) for r in train['episodes'] if r.get(key) is not None}
        b = {str(r[key]) for r in val['episodes'] if r.get(key) is not None}
        if a & b: raise ValueError('trajectory/state split overlap: ' + key)


class FrameDataset(Dataset):
    def __init__(self, index_path):
        self.index = json.loads(Path(index_path).read_text())
        if self.index.get('format') != 'shakewm.decoder.frames.v1': raise ValueError('prepared frame index required')
        self.frames = [(i, j) for i, row in enumerate(self.index['episodes']) for j, keep in enumerate(row['valid']) if keep]
        self.opened = OrderedDict()

    def __len__(self): return len(self.frames)

    def arrays(self, episode):
        if episode not in self.opened:
            row = self.index['episodes'][episode]
            self.opened[episode] = (np.load(row['features'], mmap_mode='r', allow_pickle=False),
                np.load(row['rgb'], mmap_mode='r', allow_pickle=False))
            if len(self.opened) > 8: self.opened.popitem(last=False)
        self.opened.move_to_end(episode)
        return self.opened[episode]

    def __getitem__(self, index):
        episode, frame = self.frames[index]
        z, rgb = self.arrays(episode)
        return {'latent': torch.from_numpy(np.array(z[frame], dtype=np.float32)),
            'rgb': torch.from_numpy(np.array(rgb[frame])).permute(2, 0, 1),
            'episode': episode, 'frame': frame, 'index': index}


def epoch_batches(size, batch, seed, epoch, cursor=0):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    for start in range(cursor, size, batch): yield order[start:start + batch], min(start + batch, size)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text()); inputs = cfg['inputs']
    train = prepare_split(inputs['train_manifest'], inputs['train_cache'], inputs['train_frames'], 'train', 360)
    val = prepare_split(inputs['val_manifest'], inputs['val_cache'], inputs['val_frames'], 'val', 45)
    check_split_isolation(train, val)
    print(json.dumps({'ready': True, 'train_frames': train['valid_frames'], 'val_frames': val['valid_frames']}), flush=True)


if __name__ == '__main__': main()
