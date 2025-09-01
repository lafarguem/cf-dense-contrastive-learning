from torchvision.datasets import ImageNet
from pretraining.datasets.base_dataset import CounterfactualContrastiveDataset
from torchvision.io import decode_image

class ContrastiveImageNetDataset(CounterfactualContrastiveDataset):
    def __init__(
        self,
        root_path,
        split,
        transform = None,
        cf_suffixes = [],
        multi_crop=False, 
        return_coord=False,
        target_transform = None,
        exclude_paths = [],
    ):
        super().__init__(transform, cf_suffixes, multi_crop, return_coord, target_transform)
        self.dataset = ImageNet(root_path, split, loader=decode_image)
        self.dataset.samples = [
            (path, target)
            for path, target in self.dataset.samples
            if not any(excl in path for excl in exclude_paths)
        ]
        self.targets = [s[1] for s in self.dataset.samples]
            
    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx):
        image, target = self.dataset[idx]

        sample = {}

        sample["index"] = idx
        sample["y"] = target
        if image.shape[0] == 1:
            image = image.repeat(3,1,1)
        if image.shape[0] > 3:
            image = image[0:3]
        sample["x"] = image.float() / 255.0
        return sample

    def get_counterfactual_image(self, idx, cf_suffix):
        raise NotImplementedError('ImageNet is not compatible with counterfactual generation')