from pretraining.augmentations.base_transform import BaseTransform
import torchvision.transforms as transforms
import pretraining.augmentations.utils.transform_coord as transform_coord

class InstDiscTransform(BaseTransform):
    def __init__(self, image_size, crop, normalize):
        self.normalize = transforms.Normalize(**normalize)
        self.image_size = image_size
        self.crop = crop

    def get_transform(self):
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(self.image_size, scale=(self.crop, 1.0)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            self.normalize,
        ])
        return transform, transform