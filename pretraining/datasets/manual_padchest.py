from pathlib import Path

from pretraining.datasets.base_dataset import CounterfactualContrastiveDataset
from pretraining.datasets.padchest import CounterfactualContrastivePadChestDataset

from pathlib import Path

import torch
from pretraining.datasets.caching import SharedCache
import numpy as np
from torchvision.io import decode_image
import os

class CounterfactualContrastiveManualPadChestDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        csv_path,
        manual_data_root,
        label: str,
        transform = None,
        cache: bool = False,
        max_cache_size = 24,
        cache_dim = [1,224,224],
        cf_suffixes = ['sc', 'pe', 'sc_pe'],
        multi_crop=False, 
        return_coord=False,
        cf_dir = "/vol/biomedic3/bglocker/mscproj/mal224/data/mini-padchest/padchest_cf_images_v0",
        target_transform = None,
        seg_path = None,
        parents=['scanner', 'sex', 'disease'],
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform)
        self.base_padchest = CounterfactualContrastivePadChestDataset(
            csv_path, label, transform, False, max_cache_size, cache_dim, cf_suffixes, 
            multi_crop, return_coord, cf_dir, target_transform, seg_path, None, parents
        )
        self.manual_data_root = Path(manual_data_root)
        self.names = []
        self.padchest_idxs = []
        for file in os.listdir(str(self.manual_data_root / 'images')):
            filename = os.fsdecode(file)
            idx = list(self.base_padchest.img_paths).index(filename)
            self.padchest_idxs.append(idx)
            self.names.append(filename)
        self.padchest_idxs = np.array(self.padchest_idxs)
        if cache:
            self.image_cache = SharedCache(
                size_limit_gib=max_cache_size,
                dataset_len=len(self.names),
                data_dims=cache_dim,
                dtype=torch.float32,
            )
            self.mask_cache = SharedCache(
                size_limit_gib=max_cache_size,
                dataset_len=len(self.names),
                data_dims=cache_dim,
                dtype=torch.long,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.names)

    def read_image(self, idx):
        path = self.manual_data_root / 'images' / self.names[idx]
        img = decode_image(str(path))
        if img.shape[0] == 3:  
            img = img.float()
            img = ((img[0] + img[1] + img[2]) / 2).unsqueeze(0)
        else:
            img = img.float()
        return img/(img.max() + 1e-6)  

    def read_mask(self, idx):
        path = self.manual_data_root / 'labels' / self.names[idx]
        mask = decode_image(str(path))

        out = torch.zeros(224, 224)
        
        out[mask[2] > 0] = 1
        out[mask[1] > 0] = 2

        return out.unsqueeze(0).long()

    def get_image(self, idx):
        sample = self.base_padchest.get_image(self.padchest_idxs[idx], False)
        if self.image_cache is not None:
            image = self.image_cache.get_slot(idx)
            mask = self.mask_cache.get_slot(idx)
            if image is None:
                image = self.read_image(idx)
                mask = self.read_mask(idx)
                self.image_cache.set_slot(idx, image, allow_overwrite=True)
                self.mask_cache.set_slot(idx, mask, allow_overwrite=True)
        else:
            image = self.read_image(idx)
            mask = self.read_mask(idx)
        sample['x'] = image
        sample['mask'] = mask
        
        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        return self.base_padchest.get_counterfactual_image(self.padchest_idxs[idx], cf_suffix)
    
    def _split(self, train_prop, idxs=None, balance_disease=False, **kwargs):
        if idxs is None:
            idxs = np.arange(len(self))
        
        idxs = self.padchest_idxs[idxs]
        train_idx, test_idx = self.base_padchest._split(train_prop, idxs, balance_disease, **kwargs)
        
        train_indices = np.where(np.isin(self.padchest_idxs, train_idx))[0]
        test_indices = np.where(np.isin(self.padchest_idxs, test_idx))[0]
        return train_indices, test_indices

class CounterfactualContrastivePreProcessedManualPadChestDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        csv_path,
        manual_data_root,
        label: str,
        transform = None,
        cf_suffixes = ['sc', 'pe', 'sc_pe'],
        multi_crop=False, 
        return_coord=False,
        cf_dir = "/vol/biomedic3/bglocker/mscproj/mal224/data/mini-padchest/padchest_cf_images_v0",
        target_transform = None,
        seg_path = None,
        parents=['scanner', 'sex', 'disease'],
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform)
        self.base_padchest = CounterfactualContrastivePadChestDataset(
            csv_path, label, transform, False, 0, None, cf_suffixes, 
            multi_crop, return_coord, cf_dir, target_transform, seg_path, None, parents
        )
        self.manual_data_root = Path(manual_data_root)
        self.base_images = np.load(self.manual_data_root / 'preprocessed' / 'base' / 'images.npy')
        self.base_images = torch.from_numpy(self.base_images)
        self.names = np.load(self.manual_data_root / 'preprocessed' / 'base' / 'names.npy')
        self.masks = torch.load(self.manual_data_root / 'preprocessed' / 'base' / 'masks.pt', weights_only=True)
        self.counterfactuals = {}
        for key in self.cf_suffixes:
            self.counterfactuals[key] = np.load(
                self.manual_data_root / 'preprocessed' / 'counterfactuals' / f'{key}.npy',
            )
            self.counterfactuals[key] = torch.from_numpy(self.counterfactuals[key])
        self.padchest_idxs = []
        for filename in self.names:
            idx = list(self.base_padchest.img_paths).index(filename)
            self.padchest_idxs.append(idx)
        self.padchest_idxs = np.array(self.padchest_idxs)

    def __len__(self):
        return len(self.names)

    def read_image(self, idx):
        img = self.base_images[idx]
        if img.shape[0] == 3:  
            img = img.float()
            img = ((img[0] + img[1] + img[2]) / 3).unsqueeze(0)
        else:
            img = img.float()
        return img/(img.max() + 1e-6)  

    def read_mask(self, idx):
        mask = self.masks[idx]
        return mask.long()

    def get_image(self, idx):
        sample = self.base_padchest.get_image(self.padchest_idxs[idx], False)
        image = self.read_image(idx)
        mask = self.read_mask(idx)
        sample['x'] = image
        sample['mask'] = mask
        
        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        return self.counterfactuals[cf_suffix][idx]
    
    def _split(self, train_prop, idxs=None, balance_disease=False, **kwargs):
        if idxs is None:
            idxs = np.arange(len(self))
        
        idxs = self.padchest_idxs[idxs]
        train_idx, test_idx = self.base_padchest._split(train_prop, idxs, balance_disease, **kwargs)
        
        train_indices = np.where(np.isin(self.padchest_idxs, train_idx))[0]
        test_indices = np.where(np.isin(self.padchest_idxs, test_idx))[0]

        return train_indices, test_indices