from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms
from pretraining.augmentations.utils.effects import GaussianBlur
import pretraining.augmentations.utils.transform_coord as transform_coord
from pretraining.augmentations.utils.rand_augment import rand_augment_transform

class RandAugTransform(BaseTransform):
    def __init__(self, image_size, crop, normalize):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size
        self.crop = crop

    def get_transform(self):
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(self.image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(self.image_size, scale=(self.crop, 1.0)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            rand_augment_transform('rand-n2-m10-mstd0.5', ra_params),
            transforms.RandomGrayscale(p=0.2),
            self.normalize,
        ])
        return transform, transform