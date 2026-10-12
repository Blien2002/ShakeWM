import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if device.type != 'cuda': raise ValueError('CUDA training checkpoint needs CUDA resume')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def memory(device):
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)} if device.type == 'cuda' else {}
