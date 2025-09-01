from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms
import pretraining.augmentations.utils.transform_coord as transform_coord

class NULLTransform(BaseTransform):
    def __init__(self, image_size, crop, normalize):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size
        self.crop = crop

    def get_transform(self):
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(self.image_size, scale=(self.crop, 1.0)),
            transform_coord.RandomHorizontalFlipCoord(),
            self.normalize,
        ])
        return transform, transform