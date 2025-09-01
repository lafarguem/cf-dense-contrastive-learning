from hydra.utils import instantiate
from pretraining.losses.contrastive_loss import DenseContrastiveLoss

def build_criterion(cfg, dual):
    dense_loss = instantiate(cfg.loss.dense)
    global_loss = instantiate(cfg.loss.instance)
    kwargs = {k: cfg.loss.get(k) for k in cfg.loss if k not in ['dense', 'instance']}
    criterion = DenseContrastiveLoss(dense_loss=dense_loss, global_loss=global_loss, dual_branch=dual, **kwargs)
    return criterion