from torch.utils.data import Dataset
from abc import abstractmethod, ABC
import torch
from typing import Dict, Any, Tuple, Iterable
from pretraining.utils.debug import debug_image
import time

class CounterfactualContrastiveDataset(Dataset, ABC):
    def __init__(
        self,
        transform = None,
        cf_suffixes = ['sc', 'pe'],
        multi_crop=False, 
        return_coord=False,
        target_transform=None,
    ):
        super().__init__()
        self.multi_crop = multi_crop
        self.return_coord = return_coord
        self.cf_suffixes = cf_suffixes
        self.transform = transform.get_transform() if transform is not None else None
        self.target_transform = target_transform

    @abstractmethod
    def get_image(self, idx) -> Any:
        pass

    def __getitem__(self, idx: int) -> Dict:
        return self.get(idx)
    
    def get(self, idx: int, other_transform=None, transfer=False) -> Dict:
        if other_transform is None:
            transform = self.transform
        else:
            transform = other_transform
        try:
            sample = self.get_image(idx)
            im = sample['x'].clone()
            time_ckpt = time.time()
            if transfer:
                data = transform[0](sample['x'], sample['mask']) if transform is not None else sample
                sample['x'] = data[0]
                sample['mask'] = data[1]
                transfer_time = time.time() - time_ckpt
                sample['transfer_time'] = transfer_time
                return sample
            else:
                sample = self.apply_transform(idx, sample, transform)
                transfer_time = time.time() - time_ckpt
                sample['transfer_time'] = transfer_time
                return sample
        except Exception as e:
            import traceback
            if 'sample' in locals() and 'im' in locals():
                debug_image([(sample['x'], 'x.png'), (im, 'x_og.png')])
            traceback.print_exc()
            raise e
    
    @abstractmethod
    def get_counterfactual_image(self, idx, cf_suffix) -> Any:
        pass

    def apply_transform(self, idx, sample, transform):
        cfs = [self.get_counterfactual_image(idx, cf_suffix) for cf_suffix in self.cf_suffixes]
        image = sample['x']
        mask = sample.get('mask',None)
        temp_ims = []

        if transform is not None:
            img = transform[0](image, mask)
        else:
            img = image
        temp_ims.append(img)

        if self.target_transform is not None:
            sample["y"] = self.target_transform(sample["y"])

        if self.multi_crop:
            if cfs == []:
                temp_ims.append(transform[1](image, mask))
            else:
                for cf in cfs:
                    if cf is not None:
                        temp_ims.append(transform[1](cf, mask))
                    else:
                        temp_ims.append(transform[1](image, mask))

        ims = torch.stack([temp[0] for temp in temp_ims], dim=0)  
        if mask is not None:
            masks = torch.stack([temp[1] for temp in temp_ims], dim=0)
        else:
            masks = None
        if self.return_coord:
            coords = [temp[2] for temp in temp_ims]
            sample["coords"] = coords
        sample["x"] = ims
        if masks is not None:
            sample["mask"] = masks.long()

        return sample
    
    @abstractmethod
    def _split(self, train_prop, idxs=None, **kwargs) -> Tuple[Iterable, Iterable]:
        pass

    def split(self, train_prop=0.8, eval_prop=None, idxs=None, **kwargs):
        prop = train_prop
        if eval_prop is not None:
            prop = 1 - eval_prop
        train_idx, eval_idx = self._split(prop, idxs=idxs, **kwargs)
        train_dataset = SplitDataset(self, train_idx)
        eval_dataset = SplitDataset(self, eval_idx)
        return train_dataset, eval_dataset

class SplitDataset:
    def __init__(self, dataset, idxs):
        self.base_dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        real_idx = self.idxs[idx]
        return self.base_dataset.get(real_idx)

    def get(self, idx, **kwargs):
        real_idx = self.idxs[idx]
        return self.base_dataset.get(real_idx, **kwargs)

    def __getattr__(self, name):
        if name == 'base_dataset' or name == 'idxs':
            return object.__getattribute__(self, name)
        base = object.__getattribute__(self, "base_dataset")
        return getattr(base, name)
    
    def split(self, train_prop=0.8, eval_prop=None, idxs=None, **kwargs):
        if idxs is None:
            idxs = self.idxs
        return self.base_dataset.split(train_prop=train_prop, eval_prop=eval_prop, idxs=idxs, **kwargs)

class TransferDataset:
    def __init__(self, dataset):
        self.base_dataset = dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset.get(idx, transfer=True)

    def get(self, idx, **kwargs):
        return self.base_dataset.get(idx, transfer=True, **kwargs)

    def __getattr__(self, name):
        if name == 'base_dataset':
            return object.__getattribute__(self, name)
        base = object.__getattribute__(self, "base_dataset")
        return getattr(base, name)

    def split(self, train_prop=0.8, eval_prop=None, idxs=None, **kwargs):
        split1, split2 = self.base_dataset.split(train_prop=train_prop, eval_prop=eval_prop, idxs=idxs, **kwargs)
        return TransferDataset(split1), TransferDataset(split2)

class CustomAugmentationDataset:
    def __init__(self, dataset, custom_transform):
        self.base_dataset = dataset
        self.custom_transform = custom_transform.get_transform()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset.get(idx, other_transform=self.custom_transform)

    def get(self, idx, **kwargs):
        return self.base_dataset.get(idx, other_transform=self.custom_transform, **kwargs)

    def __getattr__(self, name):
        if name == 'base_dataset' or name == 'custom_transform':
            return object.__getattribute__(self, name)
        base = object.__getattribute__(self, "base_dataset")
        return getattr(base, name)

    def split(self, train_prop=0.8, eval_prop=None, idxs=None, **kwargs):
        split1, split2 = self.base_dataset.split(train_prop=train_prop, eval_prop=eval_prop, idxs=idxs, **kwargs)
        split1_dataset = CustomAugmentationDataset(split1, self.custom_transform)
        split2_dataset = CustomAugmentationDataset(split2, self.custom_transform)
        return split1_dataset, split2_dataset

class DebugDataset:
    def __init__(self, dataset, debug_split=0.1):
        self.debug_split = debug_split
        self.base_dataset = dataset

    def __len__(self):
        return int(self.debug_split * len(self.base_dataset))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        return self.base_dataset.get(idx)

    def get(self, idx, **kwargs):
        return self.base_dataset.get(idx, **kwargs)

    def __getattr__(self, name):
        if name == 'base_dataset' or "debug_length":
            return object.__getattribute__(self, name)
        base = object.__getattribute__(self, "base_dataset")
        return getattr(base, name)

    def split(self, train_prop=0.8, eval_prop=None, idxs=None, **kwargs):
        split1, split2 = self.base_dataset.split(train_prop=train_prop, eval_prop=eval_prop, idxs=idxs, **kwargs)
        return DebugDataset(split1, self.debug_split), DebugDataset(split2, self.debug_split)