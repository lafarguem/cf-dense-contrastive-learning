from hydra.utils import instantiate

def build_criterion(cfg):
    return instantiate(cfg.loss)