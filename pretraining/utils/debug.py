from hydra.core.hydra_config import HydraConfig
import os
from torchvision.utils import save_image
import torch
import matplotlib
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

_warned_flags = set()

INCOMPATIBILITIES = [
    ({"loss.sampler.name": "random_overlap", "loss.dense._target_": ["sgd", "rmsprop"]},
     "Transformer model requires Adam or AdamW optimizer."),
    
    ({"db.backend": "sqlite", "training.distributed": True}, 
     "SQLite does not support distributed training."),
]

def get_nested(cfg, key_path):
    return OmegaConf.select(cfg, key_path)

def matches(cfg, condition):
    for key, expected in condition.items():
        actual = get_nested(cfg, key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True

def validate_config(cfg):
    for condition, message in INCOMPATIBILITIES:
        if matches(cfg, condition):
            raise ValueError(f"Incompatible config: {message}")

def check_valid_tensor(x, name=None, error=False):
    if not torch.isfinite(x).all():
        print(name, x)
        if error:
            raise RuntimeError('Invalid tensor')

def is_user_code(stat):
    for frame in stat.traceback:
        filename = frame.filename
        if 'tracemalloc' in filename:
            return False
    return True

def warn_once(logger, key, message):
    if key not in _warned_flags:
        logger.warning(message)
        _warned_flags.add(key)

def log_memory(snapshot1, snapshot2, epoch, num_lines = 10, logger=None):
    print_func = print if logger is None else logger.info
    stats_diff = snapshot2.compare_to(snapshot1, 'traceback')
    stats_diff = [stat for stat in stats_diff if is_user_code(stat)]
    total = sum([stat.size_diff for stat in stats_diff])
    positive_increases = [stat for stat in stats_diff if stat.size_diff > 0]
    print_func(f"Epoch {epoch + 1}: {total / 1024:.1f} KiB total size diff ")
    print_func(f"Epoch {epoch + 1}: Top 10 memory differences since last step")
    for i, stat in enumerate(positive_increases[:num_lines]):
        print_func(
            f"  #{i+1}: Size change = {stat.size_diff / 1024:.1f} KiB | "
            f"Count diff = {stat.count_diff}"
        )
        print_func("  Traceback (most recent call last):")
        for line in stat.traceback.format():
            print_func("    " + line)

def debug_image(images):
    debug_dir = os.path.join(HydraConfig.get().runtime.output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    for tuple in images:
        image, name = tuple
        filename = os.path.join(debug_dir, name)

        if isinstance(image, matplotlib.figure.Figure):
            image.savefig(filename)
            image.clf()
        elif isinstance(image, torch.Tensor):
            if image.dim() == 2:
                image = image.unsqueeze(0)
            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)
            save_image(image, filename)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            if image.ndim == 2:
                mode = 'L'
            elif image.shape[2] == 3:
                mode = 'RGB'
            elif image.shape[2] == 4:
                mode = 'RGBA'
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            Image.fromarray(image, mode=mode).save(filename)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

def check_index_bounds(index_tensor, dim_size):
    index_cpu = index_tensor.detach().cpu()
    print("Min index:", index_cpu.min().item())
    print("Max index:", index_cpu.max().item())
    assert (index_cpu >= 0).all(), "Index contains negative values"
    assert (index_cpu < dim_size).all(), f"Index out of bounds: max={index_cpu.max().item()}, dim={dim_size}"

def safe_gather(data: torch.Tensor, index: torch.Tensor, dim: int):
    assert index.dtype == torch.long, "Index tensor must be of dtype torch.long"

    index_cpu = index.detach().cpu()
    dim_size = data.size(dim)

    if torch.any(index_cpu < 0):
        raise ValueError(f"safe_gather: Negative indices found: min={index_cpu.min().item()}")
    if torch.any(index_cpu >= dim_size):
        raise ValueError(f"safe_gather: Index out of bounds. Max={index_cpu.max().item()}, allowed < {dim_size}")

    return torch.gather(data, dim, index)

def save_mask(labels, name):
    image = torch.zeros(3, labels.shape[-2], labels.shape[-1])
    image[0, :, :][labels[0] == 1] = 1.0
    image[1, :, :][labels[0] == 2] = 1.0
    if not os.path.exists(f'{name}.png'):
        save_image(image, f'{name}.png')

class GlobalVariableAccumulator:
    attributes = {}

    @classmethod
    def add_attribute(cls, key, value):
        cls.attributes[key] = cls.attributes.get(key, 0) + value
    
    @classmethod
    def consume_attribute(cls, key):
        val = cls.attributes.pop(key, 0)
        return val