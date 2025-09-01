from pretraining.optimizers.lars import add_weight_decay, LARS
from pretraining.utils.distributed import get_world_size
from hydra.utils import instantiate
import torch

def build_optimizer(cfg, model):
    lr = cfg.optimizer.lr
    if cfg.get("scale_lr", False):
        lr *= cfg.runtime.batch_size * get_world_size() / 256
    if 'LARS' in cfg.optimizer._target_:
        params = add_weight_decay(model, cfg.optimizer.weight_decay)
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg.optimizer.momentum,
        )
        optimizer = LARS(optimizer)
    else:
        optimizer_cfg = {
            "_target_": cfg.optimizer._target_,
            "params": model.parameters(),
            "lr": lr,
        }
        if cfg.optimizer.get('kwargs', None) is not None:
            optimizer_cfg.update(cfg.optimizer.kwargs)

        optimizer = instantiate(optimizer_cfg)

    return optimizer