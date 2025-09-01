from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pretraining.datasets.base_dataset import TransferDataset, CustomAugmentationDataset, SplitDataset
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def build_loader(cfg, logger=None):
    train_transform = instantiate(cfg.data.train.augmentation)
    train_dataset = TransferDataset(instantiate(cfg.data.train.dataset, transform=train_transform))

    if 'train_prop' in cfg.data.eval.dataset or 'eval_prop' in cfg.data.eval.dataset:
        eval_transform = instantiate(cfg.data.eval.augmentation)
        train_dataset, eval_dataset = train_dataset.split(**cfg.data.eval.dataset)
        eval_dataset = CustomAugmentationDataset(eval_dataset, eval_transform)
        if logger is not None:
            logger.info(f"length of training dataset: {len(train_dataset)}")
    elif 'num_folds' in cfg.data.eval.dataset:
        eval_transform = instantiate(cfg.data.eval.augmentation)

        kf = StratifiedGroupKFold(cfg.data.eval.dataset.num_folds)
        y = np.zeros(len(train_dataset))
        g = np.zeros(len(train_dataset))
        for idx in range(len(train_dataset)):
            s = train_dataset[idx]
            y[idx] = s['disease']
            g[idx] = s['patient']
        splits = kf.split(np.zeros(len(train_dataset)), y, g)

        train_kwargs = OmegaConf.to_container(cfg.data.train.dataloader, resolve=True)
        eval_kwargs = OmegaConf.to_container(cfg.data.eval.dataloader, resolve=True)

        all_train_dataloaders = []
        all_eval_dataloaders = []

        for split_train_idx, split_val_idx in splits:
            t_dataset, e_dataset = SplitDataset(train_dataset, split_train_idx), SplitDataset(train_dataset, split_val_idx)
            e_dataset = CustomAugmentationDataset(e_dataset, eval_transform)
            if cfg.runtime.distributed:
                train_sampler = DistributedSampler(t_dataset)
                eval_sampler = DistributedSampler(e_dataset)
            else:
                train_sampler = None
                eval_sampler = None
            all_train_dataloaders.append(DataLoader(t_dataset, sampler=train_sampler, **train_kwargs))
            all_eval_dataloaders.append(DataLoader(e_dataset, sampler=eval_sampler, **eval_kwargs))

        if logger is not None:
            logger.info(f"length of training dataset: {len(t_dataset)}")

        return all_train_dataloaders, all_eval_dataloaders
    
    else:
        if logger is not None:
            logger.info(f"length of training dataset: {len(train_dataset)}")
        eval_transform = instantiate(cfg.data.eval.augmentation)
        eval_dataset = TransferDataset(instantiate(cfg.data.eval.dataset, transform=eval_transform))

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