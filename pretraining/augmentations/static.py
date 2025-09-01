from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms

from pretraining.augmentations.utils.rand_augment import rand_augment_transform
import pretraining.augmentations.utils.transform_coord as transform_coord
from pretraining.augmentations.utils.effects import GaussianBlur, RandomSolarizeFloat

class StaticTransform(BaseTransform):
    def __init__(self, image_size, normalize):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size
    
    def get_transform(self):
        transform_1 = transform_coord.Compose([
            transform_coord.ResizeCoord(self.image_size + 32),
            transform_coord.CenterCropCoord(self.image_size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            self.normalize,
        ])
        transform_2 = transform_coord.Compose([
            transform_coord.ResizeCoord(self.image_size + 32),
            transform_coord.CenterCropCoord(self.image_size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            RandomSolarizeFloat(threshold=0.5, p=0.2),
            self.normalize,
        ])
        return transform_1, transform_2