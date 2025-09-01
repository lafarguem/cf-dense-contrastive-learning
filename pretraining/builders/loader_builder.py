from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pretraining.datasets.base_dataset import CustomAugmentationDataset, DebugDataset

def build_loader(cfg, logger=None):
    train_transform = instantiate(cfg.data.train.augmentation)
    train_dataset = instantiate(cfg.data.train.dataset, transform=train_transform)

    if 'train_prop' in cfg.data.eval.dataset or 'eval_prop' in cfg.data.eval.dataset:
        eval_transform = instantiate(cfg.data.eval.augmentation)
        train_dataset, eval_dataset = train_dataset.split(**cfg.data.eval.dataset)
        eval_dataset = CustomAugmentationDataset(eval_dataset, eval_transform)
        if logger is not None:
            logger.info(f"length of training dataset: {len(train_dataset)}")
    else:
        if logger is not None:
            logger.info(f"length of training dataset: {len(train_dataset)}")
        eval_transform = instantiate(cfg.data.eval.augmentation)
        eval_dataset = instantiate(cfg.data.eval.dataset, transform=eval_transform)

    if cfg.train.debug.debug_split is not None:
        train_dataset = DebugDataset(train_dataset, cfg.train.debug.debug_split)
        eval_dataset = DebugDataset(eval_dataset, cfg.train.debug.debug_split)

    if logger is not None:
        logger.info(f"length of evaluation dataset: {len(eval_dataset)}")

    if cfg.runtime.distributed:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    else:
        train_sampler = None
        eval_sampler = None

    train_kwargs = OmegaConf.to_container(cfg.data.train.dataloader, resolve=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **train_kwargs)

    eval_kwargs = OmegaConf.to_container(cfg.data.eval.dataloader, resolve=True)
    eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, **eval_kwargs)

    return train_loader, eval_loader