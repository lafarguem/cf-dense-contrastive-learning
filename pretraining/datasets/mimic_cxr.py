from pretraining.datasets.mimic_cxr_utils.mimic_cxr_modifiers import UIgnore, FilterBySplit, Pathology, Split, FilterByPathology
from pretraining.datasets.mimic_cxr_utils.mimic_cxr_dataset import MIMICDataset
from pretraining.datasets.base_dataset import CounterfactualContrastiveDataset
from pretraining.datasets.caching import SharedCache
from pretraining.utils.common import rle_decode
import torch
from PIL import Image
from pathlib import Path
from torchvision.io import decode_image

class CounterfactualContrastiveMimicCXRDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        root_path,
        split_path,
        split,
        label: str,
        transform = None,
        cache: bool = False,
        cf_suffixes = [],
        pathologies_only = [],
        multi_crop=False, 
        return_coord=False,
        cf_dir = "/vol/biomedic3/bglocker/mscproj/mal224/data/mini-padchest/padchest_cf_images_v0",
        meta_dir = None,
        target_transform = None,
        seg_path = None,
        transfer=False,
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform, transfer)
        self.root_path = root_path
        self.cf_dir = cf_dir
        self.label = label
        self.cache = cache
        meta_dir = Path(meta_dir)
        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None
        split = Split(split)
        self.label = Pathology(label)
        modifiers = [FilterBySplit(Split(split)), UIgnore(self.label), UIgnore(Pathology.NO_FINDING)]
        if pathologies_only:
            pathologies_only = [Pathology(pathology) for pathology in pathologies_only]
            modifiers.append(FilterByPathology(pathologies_only))
        if split == Split.TRAIN:
            meta_csv = meta_dir / 'mimic.sample.train.csv'
        elif split == Split.TEST:
            meta_csv = meta_dir / 'mimic.sample.test.csv'
        elif split == Split.VAL:
            meta_csv = meta_dir / 'mimic.sample.val.csv'
        else:
            raise ValueError('split should be one of train test val')
        self.mimic_dataset = MIMICDataset(root_path, split_path, modifiers, meta_csv, seg_path)

    def __len__(self):
        return len(self.mimic_dataset)

    def get_image(self, idx):
        if self.cache is not None:
            image = self.cache.get_slot(idx)
            if image is None:
                image, row = self.mimic_dataset[idx]
                self.cache.set_slot(idx, image, allow_overwrite=True)
            if row is None:
                row = self.mimic_dataset.get_row(idx)
        else:
            image, row = self.mimic_dataset[idx]

        sample = {}
        sample["index"] = idx
        sample["age"] = row["age"] / 100
        sample["sex"] = 0 if row['sex'] == "Male" else 1
        sample["y"] = row[self.label.value]
        sample["shortpath"] = str(row["Path"])
        sample["mask"] = rle_decode((row["Width"], row["Height"]), [row["Left Lung"], row["Right Lung"]])
        sample["x"] = image

        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        cf_dir = Path(self.cf_dir)
        short_path = self.img_paths[idx][:-4]
        filename = cf_dir / f"{short_path}_{cf_suffix}.png"
        img = decode_image(str(filename)) / 255.0
        img = img / (img.max() + 1e-12)
        return img